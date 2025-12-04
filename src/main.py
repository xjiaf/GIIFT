import sys

sys.path.append('src/models')
sys.path.append('src/utils')
import argparse
import os
import logging
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from functools import partial

from utils.multi30k import get_Multi30k
from utils.wit import get_WIT
from utils.wmt import get_WMT
# from coco_captions import *
from utils.model_utils import get_lang_code
from dataset import MultiModalDataset
from models.model import CLIPTrans, MBART
from runner import Runner
from ddp import is_main_process, init_distributed, is_dist_avail_and_initialized
from collate_fns import collate_multi

get_ds = {
    'multi30k': get_Multi30k,
    'wit': get_WIT,
    'wmt': get_WMT
}  # , 'coco': get_coco


def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logging(save_path, log_level=logging.INFO):
    """
    Set up logging to file and console.
    """

    # Create the directory if it does not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"

    # log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_format = "%(message)s"
    log_path = str(Path(save_path) / filename)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_path, mode='a', delay=True)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter(log_format))
    logger.addHandler(ch)
    return logger, filename


def main(params):
    init_seed(params.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    params.model_name = os.path.join(
        params.model_dir,
        f'{params.model_name}-{params.src_lang}-{params.tgt_lang}-{params.mapping_network}'
    )
    os.makedirs(params.model_name, exist_ok=True)
    if not params.remove_adapter:
        MODEL = CLIPTrans
    else:
        MODEL = MBART
    model = MODEL(params)
    if params.num_gpus > 1:
        init_distributed()
    else:
        available_gpus = torch.cuda.device_count()
        if params.gpu_id >= available_gpus or params.gpu_id < 0:
            print(f"GPU {params.gpu_id} not available. Using GPU 0.")
            gpu_id = 0
        else:
            gpu_id = params.gpu_id
        torch.cuda.set_device(gpu_id)
    model.cuda()
    tokenizer = model.tokenizer

    (train_texts, test_texts, train_tok, test_tok, train_image_embs,
     test_image_embs, train_text_embs, test_text_embs, train_tok_mclip,
     test_tok_mclip, merged_train_graph_list, merged_test_graph_list,
     merged_text_train_graph_list,
     merged_text_test_graph_list) = get_ds[params.ds](params, model,
                                                      params.test_ds)

    # return 0
    if params.preprocess_only:
        exit()
    train_dataset_inputs = {
        'params': params,
        'tok_data': train_tok,
        'raw_data': train_texts,
        'clip_tok_data': train_tok_mclip
    }
    test_dataset_inputs = {
        'params': params,
        'tok_data': test_tok,
        'raw_data': test_texts,
        'clip_tok_data': test_tok_mclip
    }
    stage_message = {
        'caption': 'Pretraining on Image Captions',
        'text_recon': 'Pretraining on Text Reconstruction',
        'translate': 'Pairwise Translation',
        'triplet': 'Triplet Training',
        'distill': 'Distillation'
    }
    if is_main_process():
        logger.info(stage_message[params.stage])

    train_dataset_inputs['is_pretraining'] = params.stage in [
        'caption', 'text_recon', 'distill'
    ]
    test_dataset_inputs['is_pretraining'] = params.stage in [
        'caption', 'text_recon', 'distill'
    ]

    train_dataset_inputs['mask_inputs'] = train_dataset_inputs[
        'is_pretraining'] or params.noise_train
    test_dataset_inputs['mask_inputs'] = test_dataset_inputs[
        'is_pretraining'] or params.noise_test

    if params.stage in ['text_recon', 'translate']:
        train_dataset_inputs['clip_embs'] = train_text_embs
        test_dataset_inputs[
            'clip_embs'] = test_text_embs if not params.noise_test else test_image_embs
        train_dataset_inputs['graph_data_list'] = merged_text_train_graph_list
        test_dataset_inputs['graph_data_list'] = merged_text_test_graph_list
        collate_fn = collate_multi

    elif params.stage in ['caption']:
        train_dataset_inputs['clip_embs'] = train_image_embs
        test_dataset_inputs['clip_embs'] = test_image_embs
        train_dataset_inputs['graph_data_list'] = merged_train_graph_list
        test_dataset_inputs['graph_data_list'] = merged_test_graph_list
        collate_fn = partial(collate_multi,
                             use_subgraphs=params.use_subgraphs,
                             subgraph_ratio=params.subgraph_ratio)

    elif params.stage in ['distill']:
        train_dataset_inputs['clip_embs'] = train_image_embs
        test_dataset_inputs['clip_embs'] = test_image_embs
        train_dataset_inputs['graph_data_list'] = merged_text_train_graph_list
        test_dataset_inputs['graph_data_list'] = merged_text_test_graph_list
        collate_fn = partial(collate_multi,
                             use_subgraphs=params.use_subgraphs,
                             subgraph_ratio=params.subgraph_ratio)

    elif params.stage in ['triplet']:
        train_dataset_inputs['clip_embs'] = train_image_embs
        test_dataset_inputs['clip_embs'] = test_text_embs
        train_dataset_inputs['graph_data_list'] = merged_train_graph_list
        test_dataset_inputs['graph_data_list'] = merged_text_test_graph_list
        collate_fn = partial(collate_multi,
                             use_subgraphs=params.use_subgraphs,
                             subgraph_ratio=params.subgraph_ratio)

    train_dataset = MultiModalDataset(**train_dataset_inputs)
    test_dataset = MultiModalDataset(**test_dataset_inputs)

    # if params.single_stage:
    #     # if params.caption_ds == '':
    #     #     if params.ds == 'wmt':
    #     #         params.caption_ds = 'multi30k'
    #     #     else:
    #     #         params.caption_ds = params.ds
    #     # if params.caption_ds != params.ds:
    #     #     train_texts, test_texts, train_tok, test_tok, train_image_embs, test_image_embs, train_text_embs, test_text_embs = get_ds[params.caption_ds](params, model, params.test_ds, force_pretraining = True)
    #     train_dataset_inputs['clip_embs'] = train_image_embs
    #     mix_dataset = MultiModalDataset(**train_dataset_inputs) # 'translate' -> captioning, 'triplet' -> text_recon
    #     train_dataset = ConcatDataset(train_dataset, mix_dataset)

    if not params.unfreeze_clip:
        del model.clip  # If CLIP is always frozen, we can remove it from memory since all the data is preprocessed
    elif is_main_process():
        logger.info('Also finetuning CLIP.')

    if params.num_gpus > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    if params.num_gpus > 1:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
        train_dl = DataLoader(train_dataset,
                              batch_size=params.batch_size,
                              num_workers=0,
                              collate_fn=collate_fn,
                              sampler=train_sampler)
        test_dl = DataLoader(test_dataset,
                             batch_size=2 * params.batch_size,
                             num_workers=6,
                             collate_fn=collate_fn,
                             sampler=test_sampler)
    else:
        train_dl = DataLoader(train_dataset,
                              batch_size=params.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_fn)
        test_dl = DataLoader(test_dataset,
                             batch_size=params.batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_fn)
    if is_main_process():
        if is_dist_avail_and_initialized():
            logger.info(model.module.adapter) if hasattr(
                model.module, 'adapter') else logger.info('No adapter')
        else:
            logger.info(model.adapter) if hasattr(
                model, 'adapter') else logger.info('No adapter')
        logger.info('%' * 80)
        logger.info(params)
        logger.info('%' * 80)
    runner = Runner(train_dl, test_dl, params)
    runner.train(model, tokenizer, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mn',
                        dest='model_name',
                        type=str,
                        default='',
                        help='Name of the job')
    parser.add_argument('--lm',
                        dest='load_model',
                        type=str,
                        default='',
                        help='Name of model to be loaded')
    parser.add_argument('--test',
                        action='store_true',
                        help='to run inference on a saved model')
    parser.add_argument('--ct',
                        dest='continue_training',
                        action='store_true',
                        help='flag to continue training')
    parser.add_argument('--bs', dest='batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--data_dir', type=str, default='data')
    # parser.add_argument('--data_dir', type=str, default='/mnt/iusers01/fatpou01/compsci01/n37745jx/scratch/papers/graphclip0/data')
    parser.add_argument('--src_lang', type=str, default='en')
    parser.add_argument('--tgt_lang', type=str, default='de')
    parser.add_argument(
        '--update_count',
        type=int,
        default=4,
        help='number of steps to accumulate gradient before backpropagating')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help="Don't modify, will be used automatically for DDP")
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--test_after', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--ds',
                        type=str,
                        choices=['multi30k', 'wit', 'wmt', 'coco'],
                        default='multi30k')
    parser.add_argument(
        '--unfreeze_clip',
        action='store_true',
        help='used to also finetune the CLIP text encoder in stage 2')
    parser.add_argument('--test_ds',
                        nargs='+',
                        type=str,
                        default=['2016', 'val'])
    parser.add_argument(
        '--mapping_network',
        type=str,
        default='gatl',
        choices=['gat', 'ginel', 'gcnl', 'gatl'],
        help='Choice of mapping network, refer paper for details')
    parser.add_argument('--image_encoder',
                        type=str,
                        default='mclip',
                        choices=['clip_res', 'mclip', 'clip'])
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--hidden_dims', type=int, default=300)
    parser.add_argument('--mask_prob', type=float, default=1)
    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=['caption', 'text_recon', 'translate', 'triplet', 'distill'])
    parser.add_argument('--noise_train',
                        action='store_true',
                        help='Remove mask_prob% of the tokens while training')
    parser.add_argument('--noise_test',
                        action='store_true',
                        help='Remove mask_prob% of the tokens while testing')
    parser.add_argument('--preprocess_only', action='store_true')

    # new params
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_heads', '-nh', type=int, default=8)
    parser.add_argument('--num_layers', '-nl', type=int, default=2)
    parser.add_argument(
        '--use_subgraphs',
        action='store_true',
        help='Whether to generate subgraphs during training in stage1')
    parser.add_argument(
        '--subgraph_ratio',
        type=float,
        default=0.8,
        help='The ratio of nodes to retain when generating subgraphs')
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--use_mbart_encoder',
                        action='store_true',
                        help='Use mBART encoder in the model')
    parser.add_argument('--use_fusion',
                        action='store_true',
                        help='Use fusion layer in the model')
    parser.add_argument('--use_gate',
                        action='store_true',
                        help='Use gate mechanism in the fusion layer')
    parser.add_argument('--remove_adapter',
                        action='store_true',
                        help='If using, the model will become only mBART')
    parser.add_argument('--backbone',
                        type=str,
                        default='mbart',
                        choices=['mbart', 'nllb'],
                        help='Choice of backbone for the model')

    # Enable layer-wise unfreezing
    parser.add_argument(
        '--enable_layerwise_unfreezing',
        action='store_true',
        default=False,
        help=
        'Enable layer-wise unfreezing during the pretraining phase. Default is False.'
    )

    # Set the number of epochs between each unfreeze step
    parser.add_argument(
        '--unfreeze_epoch_interval',
        type=int,
        default=None,
        help='Number of epochs between each unfreeze step.' +
        'If set to None, the unfreeze strategy is automatically calculated ' +
        'based on total training epochs and early stopping patience.')

    # Set the number of layers to unfreeze at each interval
    parser.add_argument(
        '--layers_per_unfreeze_epoch_interval',
        type=int,
        default=1,
        help='Number of layers to unfreeze at each unfreeze interval. ' +
        'Default is 1. Effective only when layer-wise unfreezing is enabled.')

    parser.add_argument('--use_src_caption',
                        action='store_true',
                        help='Use source language captions for pretraining')

    params = parser.parse_args()
    assert not (
        params.stage in ['caption', 'text_recon'] and params.ds == 'wmt'
    ), 'While using text-only NMT, you cannot train stage 1. Make sure you load a stage 1 pretrained model'
    if params.test:
        save_path = Path(
            './result', 'test',
            f"{params.model_name}-{params.mapping_network}-{params.src_lang}-{params.tgt_lang}",
            params.stage, f"test_{params.test_ds[0]}_{params.test_ds[1]}",
            f"epoch_{params.epochs}", f"mask_prob_{params.mask_prob}")
    else:
        save_path = Path(
            './result', 'train',
            f"{params.model_name}-{params.mapping_network}-{params.src_lang}-{params.tgt_lang}",
            params.stage, f"test_{params.test_ds[0]}_{params.test_ds[1]}",
            f"epoch_{params.epochs}", f"mask_prob_{params.mask_prob}")
    save_path.mkdir(parents=True, exist_ok=True)
    logger, log_filename = setup_logging(save_path)

    main(params)
    logger.info('Done!')
