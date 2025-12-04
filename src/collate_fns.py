import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from torch_geometric.data import Batch as GeometricBatch


def collate_texts(batch):  # Used with DocDataset for creating embeddings
    collate_util = lambda key, b, pad_val: pad_sequence(
        [x[key].squeeze() for x in b], batch_first=True, padding_value=pad_val)
    input_ids = collate_util('input_ids', batch, 1)
    att_mask = collate_util('attention_mask', batch, 0)
    return {'input_ids': input_ids, 'attention_mask': att_mask}


def collate_images(batch):  # Used with DocDataset for creating embeddings
    return torch.stack(batch)


def collate_multi(batch, use_subgraphs=False, subgraph_ratio=0.8):
    # Used with MultiModalDataset
    out, labels, masks = {'clip': [], 'mbart': [], 'raw': []}, [], []
    graphs = []  # Collect graph data

    for b in batch:
        # Indices for the first sample
        CLIP_IDX = 0
        MBART_IDX = 1
        MASK_IDX = 2
        LABEL_IDX = 3
        RAW_IDX = 4
        GRAPH_IDX = 5

        # First sample
        out['clip'].append(b[CLIP_IDX])
        out['mbart'].append(b[MBART_IDX])
        masks.append(b[MASK_IDX])
        labels.append(b[LABEL_IDX])
        out['raw'].append(b[RAW_IDX])
        graphs.append(b[GRAPH_IDX])  # Graph data of the first sample

        if len(b) > GRAPH_IDX + 1:  # If there is a second sample
            # Indices for the second sample
            SECOND_OFFSET = GRAPH_IDX + 1  # Start index for the second sample
            out['clip'].append(b[SECOND_OFFSET + CLIP_IDX])
            out['mbart'].append(b[SECOND_OFFSET + MBART_IDX])
            masks.append(b[SECOND_OFFSET + MASK_IDX])
            labels.append(b[SECOND_OFFSET + LABEL_IDX])
            out['raw'].append(b[SECOND_OFFSET + RAW_IDX])
            graphs.append(b[SECOND_OFFSET +
                            GRAPH_IDX])  # Graph data of the second sample

    # Process graph data
    if all(graph is not None for graph in graphs):
        if use_subgraphs:
            graphs = [generate_subgraph(graph, subgraph_ratio) for graph in graphs]
        graph_batch = GeometricBatch.from_data_list(graphs)
    else:
        graph_batch = None

    # Process other data
    out['clip'] = collate_images(out['clip']) if not isinstance(
        out['clip'][0],
        transformers.tokenization_utils_base.BatchEncoding) else collate_texts(
            out['clip'])
    out['mbart'] = collate_texts(out['mbart'])
    out['mbart']['labels'] = collate_texts(labels)['input_ids']
    out['mbart']['mask_decoder_input_ids'] = pad_sequence(
        [x.squeeze() for x in masks], batch_first=True, padding_value=True)

    # Include graph_batch within out
    out['graph'] = graph_batch

    return out  # Return the batched data including graph_batch


def generate_subgraph(graph, subgraph_ratio):
    num_nodes = graph.num_nodes
    num_sub_nodes = max(1, int(num_nodes * subgraph_ratio))
    subset = torch.randperm(num_nodes)[:num_sub_nodes]
    subset = subset.sort()[0]

    # Generate subgraph data
    subgraph_data = graph.subgraph(subset)

    return subgraph_data
