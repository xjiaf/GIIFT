import sys
import torch
import json
from tqdm import tqdm
import pickle as pkl
from PIL import Image
import os
from model_utils import get_lang_code, send_to_cuda
from collate_fns import *
from dataset import DocDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data


def get_image_embs(clip,
                   folder,
                   image_splits,
                   image_embs_f,
                   desc,
                   preprocessor,
                   ignore_indices=[]):
    try:
        img_embs = torch.load(image_embs_f)
    except:
        img_paths = [os.path.join(folder, f) for f in image_splits]
        imgs = []
        for i, path in enumerate(tqdm(img_paths, desc='Reading images')):
            if i in ignore_indices:
                continue
            if '#' in path:
                path = path[:path.index(
                    '#')]  # For multi30k mscoco test splits
            try:
                img = read_image(path, preprocessor)
                imgs.append(img)
            except:
                ignore_indices.append(i)
        img_ds = DocDataset(imgs)
        img_dl = DataLoader(img_ds,
                            batch_size=128,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_images)
        img_embs = create_embeddings(img_dl, clip, image_embs_f, desc)
    return img_embs


def read_image(img_path, preprocessor):
    img = Image.open(img_path)
    try:
        img = torch.from_numpy(preprocessor(images=img)['pixel_values']
                               [0])  # For compatibility with Huggingface
    except:
        img = preprocessor(img)  # For compatibility with openai clip

    return img


def tokenize(texts, tokenizer, lang, outfile, backbone, desc=''):
    if outfile is None:
        data = []
        tokenizer.src_lang = get_lang_code(lang, backbone)
        for text in texts:
            data.append(tokenizer(text, return_tensors='pt', truncation=True))
        return data
    elif not os.path.exists(outfile):
        data = []
        tokenizer.src_lang = get_lang_code(lang, backbone)
        for text in tqdm(texts, desc=desc):
            data.append(tokenizer(text, return_tensors='pt', truncation=True))

        with open(outfile, 'wb') as f:
            pkl.dump(data, f)
    else:
        with open(outfile, 'rb') as f:
            data = pkl.load(f)
    return data


def create_embeddings(dl, encoder, outfile=None, desc=''):
    embs = torch.tensor([]).cuda()
    with torch.no_grad():
        with torch.autocast(device_type='cuda'):
            if outfile is not None:
                for data in tqdm(dl, desc=desc):
                    embs = torch.cat((embs, encoder(send_to_cuda(data))),
                                     dim=0)
            else:
                for data in dl:
                    embs = torch.cat((embs, encoder(send_to_cuda(data))),
                                     dim=0)
    if outfile is not None:
        torch.save(embs.cpu(), outfile)
    return embs


def postprocess_pairs(train_texts, test_texts, train_tok_mbart, test_tok_mbart,
                      train_img_embs, test_img_embs, train_text_embs,
                      test_text_embs, params, train_ignore_indices,
                      test_ignore_indices, train_tok_mclip, test_tok_mclip,
                      force_pretraining):
    if params.stage in [
            'caption', 'text_recon'
    ] or force_pretraining:  # Need paired images in this stage, use ignore_indices to remove the respective test indices. force_pretraining used for caption_dataset in stage 4
        for index in sorted(
                train_ignore_indices[::-1], reverse=True
        ):  # Reversing before sorting since most of the list is already sorted in ascending order -> faster sorting
            for lang in train_texts.keys():
                train_texts[lang].pop(index)
                train_tok_mbart[lang].pop(index)

        train_mask = torch.isin(torch.arange(
            train_text_embs[params.src_lang].shape[0]),
                                torch.tensor(train_ignore_indices),
                                invert=True)
        for lang in train_texts.keys():
            train_text_embs[lang] = train_text_embs[lang][train_mask]

        for index in sorted(
                test_ignore_indices[::-1], reverse=True
        ):  # Reversing before sorting since most of the list is already sorted in ascending order -> faster sorting
            for lang in test_texts.keys():
                test_texts[lang].pop(index)
                test_tok_mbart[lang].pop(index)

        test_mask = torch.isin(torch.arange(
            test_text_embs[params.src_lang].shape[0]),
                               torch.tensor(test_ignore_indices),
                               invert=True)
        for lang in test_texts.keys():
            test_text_embs[lang] = test_text_embs[lang][test_mask]

        # Stage 2 and 3 dont use image embs so they can use the complete text
        for lang in train_texts.keys():
            assert len(train_texts[lang]) == len(train_tok_mbart[lang]) == len(
                train_text_embs[lang]), 'Misalignment in train text pairs'
            if params.stage == 1:
                assert len(train_texts[lang]) == len(
                    train_tok_mbart[lang]) == len(
                        train_text_embs[lang]) == len(
                            train_img_embs
                        ), 'Misalignment in train text pairs with images'
        for lang in test_texts.keys():
            assert len(test_texts[lang]) == len(test_tok_mbart[lang]) == len(
                test_text_embs[lang]), 'Misalignment in train text pairs'
            if params.stage == 1:
                assert len(test_texts[lang]) == len(
                    test_tok_mbart[lang]) == len(test_text_embs[lang]) == len(
                        test_img_embs
                    ), 'Misalignment in test text pairs with images'

    return (train_texts, test_texts, train_tok_mbart, test_tok_mbart, train_img_embs, test_img_embs,
        train_text_embs, test_text_embs, train_tok_mclip, test_tok_mclip)


def process_scene_graphs(jsonl_file, model):
    """
    Processes scene graphs and returns a list of PyG Data objects.

    Parameters:
    - jsonl_file: Path to the JSONL file containing scene graphs.
    - model: The model used for embedding entities and predicates.
    - graph_type: 'image' for image scene graphs, 'text' for text scene graphs.

    Returns:
    - data_list: List of PyG Data objects.
    """
    data_list = []

    with open(jsonl_file, 'r') as f:
        # Read the first line to determine the key name
        first_line = json.loads(f.readline().strip())
        key_name = next((key for key in first_line if 'scene_graph' in key),
                        None)
        if key_name is None:
            raise KeyError(
                "No key containing 'scene_graph' found in the first line of the data."
            )

        # Rewind the file pointer back to the beginning to process all lines
        f.seek(0)

        pbar = tqdm(desc='Processing scene graphs',
                    file=sys.stdout,
                    unit='line',
                    total=None)
        for line in f:
            # Parse the JSON line
            line_data = json.loads(line.strip())
            image_name = line_data['image_name']
            scene_graph_str = line_data[key_name]

            # The scene_graph is a string containing a JSON object, so parse it
            scene_graph = json.loads(scene_graph_str)

            entities = scene_graph['entities']  # List of entity strings
            relations = scene_graph['relations']  # List of relations

            # Map entities to indices (only needed for 'image' graph_type)
            entity_to_idx = {
                entity: idx
                for idx, entity in enumerate(entities)
            }

            # Prepare x: embeddings of entities
            # Tokenize entities
            entity_tokens = tokenize(entities,
                                     model.clip.text_preprocessor,
                                     lang='en',
                                     outfile=None,
                                     backbone=model.backbone,
                                     desc='Tokenizing entities')

            # Create a DataLoader for entities
            entity_ds = DocDataset(entity_tokens)
            entity_dl = DataLoader(entity_ds,
                                   batch_size=256,
                                   shuffle=False,
                                   num_workers=0,
                                   collate_fn=collate_texts)

            # Create embeddings for entities
            entity_embeddings = create_embeddings(entity_dl,
                                                  model.clip,
                                                  outfile=None,
                                                  desc='')

            x = entity_embeddings  # Should be a tensor of size [num_entities, embedding_dim]

            # Prepare edge_index and edge_attr
            edge_index = []
            edge_attr_texts = []

            for rel in relations:
                subject_str, predicate, object_str = rel
                # Get indices of subject and object
                if subject_str in entity_to_idx and object_str in entity_to_idx:
                    subj_idx = entity_to_idx[subject_str]
                    obj_idx = entity_to_idx[object_str]
                else:
                    raise ValueError(
                        f"Entity not found in entity_to_idx: {subject_str}, {object_str}"
                    )

                edge_index.append([subj_idx, obj_idx])
                edge_attr_texts.append(predicate)

            if len(edge_index) == 0:
                # No edges, skip this graph
                continue

            edge_index = torch.tensor(
                edge_index,
                dtype=torch.long).t().contiguous()  # shape [2, num_edges]

            # Tokenize predicates
            predicate_tokens = tokenize(edge_attr_texts,
                                        model.clip.text_preprocessor,
                                        lang='en',
                                        outfile=None,
                                        backbone=model.backbone,
                                        desc='Tokenizing predicates')

            # Create DataLoader for predicates
            predicate_ds = DocDataset(predicate_tokens)
            predicate_dl = DataLoader(predicate_ds,
                                      batch_size=256,
                                      shuffle=False,
                                      num_workers=0,
                                      collate_fn=collate_texts)

            # Create embeddings for predicates
            predicate_embeddings = create_embeddings(predicate_dl,
                                                     model.clip,
                                                     outfile=None,
                                                     desc='')

            edge_attr = predicate_embeddings  # Should be tensor of size [num_edges, embedding_dim]

            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            # Optionally, you can add other attributes, e.g., image_name
            data.image_name = image_name

            data_list.append(data)
            pbar.update(1)
        else:
            pbar.close()

    return data_list


def merge_graphs(data1: Data, data2: Data) -> Data:
    num_nodes1 = data1.num_nodes

    # Clone data2 to avoid modifying the original data
    data2 = data2.clone()

    # Adjust the edge indices of data2
    data2.edge_index = data2.edge_index + num_nodes1

    # Combine node features
    x_combined = torch.cat([data1.x, data2.x], dim=0)

    # Combine edge indices
    edge_index_combined = torch.cat([data1.edge_index, data2.edge_index],
                                    dim=1)

    # Create a new Data object
    data_combined = Data(x=x_combined, edge_index=edge_index_combined)

    # Combine edge attributes if they exist
    if data1.edge_attr is not None and data2.edge_attr is not None:
        edge_attr_combined = torch.cat([data1.edge_attr, data2.edge_attr],
                                       dim=0)
        data_combined.edge_attr = edge_attr_combined

    return data_combined
