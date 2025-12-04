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
from multi30k import (load_or_create_embeddings, merge_and_save_graphs,
                      load_or_create_scene_graph_list,
                      load_or_create_tokenized_data)

logger = logging.getLogger("__main__")


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
        if merge_type == 'text':
            merged_graph_dataset_file = os.path.join(
                datapath, 'text_scene_graphs',
                f'merged_text_{mode}_{params.tgt_lang}_{params.src_lang}_{params.image_encoder}.pth'
            )
            graph_dataset_file = None  # No image scene graph
            graph_jsonl_file = None
        else:
            raise ValueError(f'Invalid merge_type: {merge_type}')

        text_graph_dataset_file = os.path.join(
            datapath, 'text_scene_graphs',
            f'text_{mode}_{params.tgt_lang}_{params.src_lang}_{params.image_encoder}.pth'
        )
        text_graph_jsonl_file = os.path.join(
            datapath, 'text_scene_graphs',
            f'text_{mode}_{params.tgt_lang}_{params.src_lang}.jsonl')
    elif mode == 'test':
        test0, test1 = test_info
        if merge_type == 'text':
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


def get_WMT(params, model, test=('2014', 'wmt')):
    langs = [params.src_lang, params.tgt_lang]
    # if test on wmt
    if 'wmt' in test[1]:
        test[1] = f'wmt_{params.tgt_lang}_{params.src_lang}'
    elif 'val' in test[1]:
        test[1] = f'val_{params.tgt_lang}_{params.src_lang}'

    datapath = os.path.join(params.data_dir, 'wmt')
    backbone = params.backbone
    os.makedirs(os.path.join(datapath, f'text/data/task1/{backbone}'),
                exist_ok=True)
    os.makedirs(os.path.join(datapath,
                             f'text/data/task1/{params.image_encoder}'),
                exist_ok=True)

    # Reading train files
    train_texts = {
        lang:
        open(
            os.path.join(
                datapath,
                f'text/data/task1/raw/train_{params.tgt_lang}_{params.src_lang}.{lang}'
            )).read().splitlines()
        for lang in langs
    }

    # Load or create tokenized data for backbone
    train_tok_backbone = {}
    for lang in langs:
        filepath = os.path.join(
            datapath,
            f'text/data/task1/{backbone}/train_{params.tgt_lang}_{params.src_lang}.{lang}.pkl'
        )
        description = f'Tokenizing train {lang} with backbone'
        train_tok_backbone[lang] = load_or_create_tokenized_data(
            train_texts[lang], model.tokenizer, lang, filepath, backbone, description)

    # Load or create tokenized data for mclip
    train_tok_mclip = {}
    for lang in langs:
        filepath = os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/train_{params.tgt_lang}_{params.src_lang}.{lang}.pkl'
        )
        description = f'Tokenizing train {lang} with {params.image_encoder}'
        train_tok_mclip[lang] = load_or_create_tokenized_data(
            train_texts[lang], model.clip.text_preprocessor, lang, filepath, backbone,
            description)

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
            test_texts[lang], model.tokenizer, lang, filepath, backbone, description)

    # Load or create tokenized test data for mclip
    test_tok_mclip = {}
    for lang in langs:
        filepath = os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.{lang}.pkl'
        )
        description = f'Tokenizing test {lang} with {params.image_encoder}'
        test_tok_mclip[lang] = load_or_create_tokenized_data(
            test_texts[lang], model.clip.text_preprocessor, lang, filepath, backbone,
            description)

    # Reading image splits
    # train_image_splits = open(
    #     os.path.join(
    #         datapath,
    #         f'text/data/task1/image_splits/train_{params.tgt_lang}_{params.src_lang}.txt')).read().splitlines()
    # test_image_splits = open(
    #     os.path.join(
    #         datapath,
    #         f'text/data/task1/image_splits/test_{test[0]}_{test[1]}.txt')
    # ).read().splitlines()

    # Getting images and embedding with CLIP
    # logger.info('Loaded all text files. Getting images...')
    train_img_embs = None
    test_img_embs = None
    # train_img_embs = load_or_create_image_embeddings(
    #     model.clip, os.path.join(datapath, 'images/train'), train_image_splits,
    #     os.path.join(datapath,
    #                  f'text/data/task1/{params.image_encoder}/train.pth'),
    #     'Embedding train images', model.clip.image_preprocessor)

    # test_img_embs = load_or_create_image_embeddings(
    #     model.clip, os.path.join(datapath, f'images/test_{test[0]}_{test[1]}'),
    #     test_image_splits,
    #     os.path.join(
    #         datapath,
    #         f'text/data/task1/{params.image_encoder}/test_{test[0]}_{test[1]}.pth'
    #     ), f'Embedding test_{test[0]}_{test[1]} images',
    #     model.clip.image_preprocessor)

    # Load or create text embeddings
    train_text_embs = {}
    test_text_embs = {}
    for lang in langs:
        # Train text embeddings
        embs_f_train = os.path.join(
            datapath,
            f'text/data/task1/{params.image_encoder}/train_{params.tgt_lang}_{params.src_lang}.{lang}.pth'
        )
        description_train = f'Embedding train_{params.tgt_lang}_{params.src_lang}.{lang} {params.image_encoder}'
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
    merged_train_graph_list = None

    # Process and merged test multimodel scene graphs
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
