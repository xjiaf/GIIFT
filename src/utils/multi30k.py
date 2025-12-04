import sys

sys.path.append('..')
import torch
from tqdm import tqdm
import logging
import pickle as pkl
import os
from data_utils import *
from collate_fns import *
from dataset import DocDataset
from torch.utils.data import DataLoader
from ddp import *
from super_node import SuperNode

logger = logging.getLogger("__main__")


def load_or_create_tokenized_data(texts, tokenizer, lang, filepath, backbone,
                                  description):
    try:
        data = pkl.load(open(filepath, 'rb'))
        logger.info(f'Loaded tokenized data for {lang} from {filepath}')
    except OSError:
        logger.info(f'Did not find tokenized data for {lang}. Creating...')
        data = tokenize(texts, tokenizer, lang, filepath, backbone, description)
    return data


def load_or_create_embeddings(data_loader, model, filepath, description):
    try:
        embs = torch.load(filepath)
        logger.info(f'Loaded embeddings from {filepath}')
    except OSError:
        logger.info(f'Did not find embeddings. Creating...')
        embs = create_embeddings(data_loader, model, filepath, description)
    return embs


def load_or_create_image_embeddings(model, image_dir, image_splits, filepath,
                                    description, image_preprocessor):
    try:
        embs = torch.load(filepath)
        logger.info(f'Loaded image embeddings from {filepath}')
    except OSError:
        logger.info(f'Did not find image embeddings. Creating...')
        embs = get_image_embs(model, image_dir, image_splits, filepath,
                              description, image_preprocessor)
    return embs


def load_or_create_scene_graph_list(filepath, jsonl_file, model, description):
    try:
        data_list = torch.load(filepath)
        logger.info(f'Loaded scene graph data from {filepath}')
        return data_list
    except OSError:
        if jsonl_file is None:
            logger.error(
                f'JSONL file is None and {filepath} does not exist. Cannot create scene graph data.'
            )
            return None
        logger.info(f'Did not find scene graph data. Creating...')
        data_list = process_scene_graphs(jsonl_file, model)  # on GPU
        torch.save([data.cpu() for data in data_list], filepath)
        return data_list


def merge_and_save_graphs(img_graph_list, txt_graph_list, super_features,
                          output_filepath, description):
    logger.info(description)
    merged_graph_list = []
    super_features = super_features.cpu()
    transform = SuperNode(super_features=super_features)
    if img_graph_list is not None and txt_graph_list is not None:
        img_graph_list = [data.cpu() for data in img_graph_list]
        txt_graph_list = [data.cpu() for data in txt_graph_list]
        for idx in tqdm(range(len(img_graph_list)), desc=description):
            data1 = img_graph_list[idx]
            data2 = txt_graph_list[idx]

            # Merge the two graphs
            merged_data = merge_graphs(data1, data2)

            # Apply the SuperNode transform
            merged_data = transform(merged_data, idx=idx)

            merged_graph_list.append(merged_data.cpu())
    else:
        txt_graph_list = [data.cpu() for data in txt_graph_list]
        for idx in tqdm(range(len(txt_graph_list)), desc=description):
            data1 = txt_graph_list[idx]

            # Apply the SuperNode transform
            data1 = transform(data1, idx=idx)

            merged_graph_list.append(data1.cpu())

    torch.save([data.cpu() for data in merged_graph_list], output_filepath)
    logger.info(f'Saved merged graph data to {output_filepath}')
    return merged_graph_list


def process_and_merge_scene_graphs(mode,
                                   params,
                                   model,
                                   super_features,
                                   datapath,
                                   test_info=None,
                                   merge_type='multi'):
    """
    Processes and merges scene graphs.

    Parameters:
    - mode: 'train' or 'test'
    - params: parameters including image_encoder
    - model: the model used for processing
    - super_features: features for the SuperNode (e.g., image embeddings or text embeddings)
    - datapath: base data path
    - test_info: tuple containing test dataset identifiers
    - merge_type: 'multi' for merging image and text scene graphs,
                  'text' for merging text scene graph with super_features (text embeddings)
    """
    if mode == 'train':
        if merge_type == 'multi':
            merged_graph_dataset_file = os.path.join(
                datapath, 'scene_graphs',
                f'merged_{mode}_{params.image_encoder}.pth')
            graph_dataset_file = os.path.join(
                datapath, 'scene_graphs', f'{mode}_{params.image_encoder}.pth')
            graph_jsonl_file = os.path.join(datapath, 'scene_graphs',
                                            f'{mode}.jsonl')
        elif merge_type == 'text':
            merged_graph_dataset_file = os.path.join(
                datapath, 'text_scene_graphs',
                f'merged_text_{mode}_{params.image_encoder}.pth')
            graph_dataset_file = None  # No image scene graph
            graph_jsonl_file = None
        else:
            raise ValueError(f'Invalid merge_type: {merge_type}')

        text_graph_dataset_file = os.path.join(
            datapath, 'text_scene_graphs',
            f'text_{mode}_{params.image_encoder}.pth')
        text_graph_jsonl_file = os.path.join(datapath, 'text_scene_graphs',
                                             f'text_{mode}.jsonl')
    elif mode == 'test':
        test0, test1 = test_info
        if merge_type == 'multi':
            merged_graph_dataset_file = os.path.join(
                datapath, 'scene_graphs',
                f'merged_{mode}_{test0}_{test1}_{params.image_encoder}.pth')
            graph_dataset_file = os.path.join(
                datapath, 'scene_graphs',
                f'{mode}_{test0}_{test1}_{params.image_encoder}.pth')
            graph_jsonl_file = os.path.join(datapath, 'scene_graphs',
                                            f'{mode}_{test0}_{test1}.jsonl')
        elif merge_type == 'text':
            merged_graph_dataset_file = os.path.join(
                datapath, 'text_scene_graphs',
                f'merged_text_{mode}_{test0}_{test1}_{params.image_encoder}.pth'
            )
            graph_dataset_file = None  # No image scene graph
            graph_jsonl_file = None
        else:
            raise ValueError(f'Invalid merge_type: {merge_type}')

        text_graph_dataset_file = os.path.join(
            datapath, 'text_scene_graphs',
            f'text_{mode}_{test0}_{test1}_{params.image_encoder}.pth')
        text_graph_jsonl_file = os.path.join(
            datapath, 'text_scene_graphs',
            f'text_{mode}_{test0}_{test1}.jsonl')
    else:
        raise ValueError(f'Invalid mode: {mode}')

    merged_graph_list = None
    try:
        merged_graph_list = torch.load(merged_graph_dataset_file)
        logger.info(
            f'Loaded merged {mode} scene graph data from {merged_graph_dataset_file}'
        )
    except (OSError, FileNotFoundError):
        logger.info(f'Merged {mode} scene graph data not found. Creating...')
        # Load or create individual scene graphs
        if graph_dataset_file is not None:
            graph_list = load_or_create_scene_graph_list(
                graph_dataset_file, graph_jsonl_file, model,
                f'{mode} scene graph data')
        else:
            graph_list = None  # No image scene graph

        text_graph_list = load_or_create_scene_graph_list(
            text_graph_dataset_file, text_graph_jsonl_file, model,
            f'text {mode} scene graph data')

        # Merge and save
        merged_graph_list = merge_and_save_graphs(
            graph_list, text_graph_list, super_features,
            merged_graph_dataset_file,
            f'Merging {mode} scene graphs with merge_type={merge_type}')
    return merged_graph_list


def get_Multi30k(params,
                 model,
                 test=('2017', 'mscoco'),
                 force_pretraining=False):
    if force_pretraining:
        langs = ['en']  # Only pretraining on en images
        test = ('2017', 'mscoco')  # Anyway not going to be used
    else:
        langs = [params.src_lang, params.tgt_lang]
    # if test on wmt
    if 'wmt' in test[1]:
        test[1] = f'wmt_{params.tgt_lang}_{params.src_lang}'
    backbone = params.backbone
    image_datapath = os.path.join(params.data_dir, 'multi30k')

    if "uk" in langs:
        logger.info('Using Ukrainian data')
        datapath = os.path.join(params.data_dir, 'multi30k_uk')
    else:
        datapath = os.path.join(params.data_dir, 'multi30k')
    os.makedirs(os.path.join(datapath, f'text/data/task1/{backbone}'),
                exist_ok=True)
    os.makedirs(os.path.join(datapath,
                             f'text/data/task1/{params.image_encoder}'),
                exist_ok=True)

    # Reading train files
    train_texts = {
        lang:
        open(os.path.join(
            datapath,
            f'text/data/task1/raw/train.{lang}')).read().splitlines()
        for lang in langs
    }

    # Load or create tokenized data for backbone
    train_tok_backbone = {}
    for lang in langs:
        filepath = os.path.join(
            datapath, f'text/data/task1/{backbone}/train.{lang}.pkl')
        description = f'Tokenizing train {lang} with backbone'
        train_tok_backbone[lang] = load_or_create_tokenized_data(
            train_texts[lang], model.tokenizer, lang, filepath, backbone,
            description)

    # Load or create tokenized data for mclip
    train_tok_mclip = {}
    for lang in langs:
        filepath = os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/train.{lang}.pkl')
        description = f'Tokenizing train {lang} with {params.image_encoder}'
        train_tok_mclip[lang] = load_or_create_tokenized_data(
            train_texts[lang], model.clip.text_preprocessor, lang, filepath,
            backbone, description)

    # Reading test files
    test_texts = {
        lang:
        open(
            os.path.join(
                datapath,
                f'text/data/task1/raw/test_{test[0]}_{test[1]}.{lang}')).read(
                ).splitlines()
        for lang in langs
    }

    # Load or create tokenized test data for backbone
    test_tok_backbone = {}
    for lang in langs:
        filepath = os.path.join(
            datapath,
            f'text/data/task1/{backbone}/test_{test[0]}_{test[1]}.{lang}.pkl')
        description = f'Tokenizing test {lang} with backbone'
        test_tok_backbone[lang] = load_or_create_tokenized_data(
            test_texts[lang], model.tokenizer, lang, filepath, backbone,
            description)

    # Load or create tokenized test data for mclip
    test_tok_mclip = {}
    for lang in langs:
        filepath = os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pkl'
        )
        description = f'Tokenizing test {lang} with {params.image_encoder}'
        test_tok_mclip[lang] = load_or_create_tokenized_data(
            test_texts[lang], model.clip.text_preprocessor, lang, filepath,
            backbone, description)

    # Reading image splits
    train_image_splits = open(
        os.path.join(
            datapath,
            f'text/data/task1/image_splits/train.txt')).read().splitlines()
    test_image_splits = open(
        os.path.join(
            datapath,
            f'text/data/task1/image_splits/test_{test[0]}_{test[1]}.txt')
    ).read().splitlines()

    # Getting images and embedding with CLIP
    logger.info('Loaded all text files. Getting images...')
    train_img_embs = load_or_create_image_embeddings(
        model.clip, os.path.join(image_datapath, 'images/train'), train_image_splits,
        os.path.join(datapath,
                     f'text/data/task1/{params.image_encoder}/train.pth'),
        'Embedding train images', model.clip.image_preprocessor)

    if "uk" in langs and 'val' in test:
        image_file = os.path.join(image_datapath, f'images/train')
    else:
        image_file = os.path.join(image_datapath, f'images/test_{test[0]}_{test[1]}')
    test_img_embs = load_or_create_image_embeddings(
        model.clip, image_file,
        test_image_splits,
        os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.pth'
        ), f'Embedding test_{test[0]}_{test[1]} images',
        model.clip.image_preprocessor)

    # Load or create text embeddings
    train_text_embs = {}
    test_text_embs = {}
    for lang in langs:
        # Train text embeddings
        embs_f_train = os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/train.{lang}.pth')
        description_train = f'Embedding train.{lang} {params.image_encoder}'
        text_ds_train = DocDataset(train_tok_mclip[lang])
        text_dl_train = DataLoader(text_ds_train,
                                   batch_size=256,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_texts)
        train_text_embs[lang] = load_or_create_embeddings(
            text_dl_train, model.clip, embs_f_train, description_train)

        # Test text embeddings
        embs_f_test = os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pth'
        )
        description_test = f'Embedding test_{test[0]}_{test[1]}.{lang} {params.image_encoder}'
        text_ds_test = DocDataset(test_tok_mclip[lang])
        text_dl_test = DataLoader(text_ds_test,
                                  batch_size=256,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=collate_texts)
        test_text_embs[lang] = load_or_create_embeddings(
            text_dl_test, model.clip, embs_f_test, description_test)

    # Process and merged train multimodel scene graphs
    if params.stage in ['caption', 'distill']:
        merged_train_graph_list = process_and_merge_scene_graphs(
            mode='train',
            params=params,
            model=model,
            super_features=train_img_embs,
            datapath=datapath,
            merge_type='multi')
    else:
        merged_train_graph_list = None

    # Process and merged test multimodel scene graphs
    if params.stage in ['caption', 'distill']:
        merged_test_graph_list = process_and_merge_scene_graphs(
            mode='test',
            params=params,
            model=model,
            super_features=test_img_embs,
            datapath=datapath,
            test_info=test,
            merge_type='multi')
    else:
        merged_test_graph_list = None

    # Process and merged text test scene graphs
    merged_text_train_graph_list = process_and_merge_scene_graphs(
        mode='train',
        params=params,
        model=model,
        super_features=train_text_embs['en'],  # Parser only supports en
        datapath=datapath,
        merge_type='text')

    # Process and merged text test scene graphs
    merged_text_test_graph_list = process_and_merge_scene_graphs(
        mode='test',
        params=params,
        model=model,
        super_features=test_text_embs['en'],  # Parser only supports en
        datapath=datapath,
        test_info=test,
        merge_type='text')

    return (train_texts, test_texts, train_tok_backbone, test_tok_backbone,
            train_img_embs, test_img_embs, train_text_embs, test_text_embs,
            train_tok_mclip, test_tok_mclip, merged_train_graph_list,
            merged_test_graph_list, merged_text_train_graph_list,
            merged_text_test_graph_list)
