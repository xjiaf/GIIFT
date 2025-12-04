import sys

sys.path.append('src/utils')
import torch.nn as nn
import logging

# Import both MBART components and general Auto* classes
from transformers import (MBart50Tokenizer, MBartForConditionalGeneration,
                          AutoTokenizer, AutoModelForSeq2SeqLM)
from transformers.modeling_outputs import BaseModelOutput

from clip_comb import CLIP, M_CLIP, CLIP_RES
from adapters import (GATAdapter, GraphAdapterLarge, GATAdapterLarge,
                      GINEAdapterLarge)
from utils.model_utils import get_lang_code, freeze_params

logger = logging.getLogger("__main__")


class CLIPTrans(nn.Module):
    """
    This class can work with either MBART or NLLB as the backbone, controlled by `params.backbone`.
    The parameter `use_backbone_encoder` indicates whether to run the original encoder
    (MBART or NLLB) plus the adapter, or only the adapter.
    """

    def __init__(self, params):
        super(CLIPTrans, self).__init__()
        self.params = params
        if params.image_encoder == 'mclip':
            logger.info('Using MCLIP encoder')
            self.clip = M_CLIP(params)
        elif params.image_encoder == 'clip':
            logger.info('Using CLIP encoder')
            self.clip = CLIP(params)
        elif params.image_encoder == 'clip_res':
            logger.info('Using CLIP-Resnet encoder')
            self.clip = CLIP_RES(params)

        # Choose backbone: 'mbart' or 'nllb'
        self.backbone = params.backbone  # 'mbart' or 'nllb'
        self.use_backbone_encoder = params.use_mbart_encoder  # Renamed logic from "use_mbart_encoder" to "use_backbone_encoder"
        self.use_fusion = params.use_fusion

        if self.backbone == 'mbart':
            # MBART-based backbone
            logger.info("Backbone set to MBART")
            self.tokenizer = MBart50Tokenizer.from_pretrained(
                "facebook/mbart-large-50")
            self.backbone_model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50")
            # MBART-50 has hidden_dim = 1024
            self.encoder_hidden_dim = 1024

        elif self.backbone == 'nllb':
            # NLLB-based backbone
            logger.info(
                "Backbone set to NLLB (facebook/nllb-200-distilled-600M)")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/nllb-200-distilled-600M")
            self.backbone_model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/nllb-200-distilled-600M")
            # The hidden_dim of nllb-200-distilled-600M is also 1024
            self.encoder_hidden_dim = 1024

        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        # Freeze the encoder if needed
        if not params.test:
            if params.stage in ['caption', 'text_recon', 'distill']:
                freeze_params(self.backbone_model.model.encoder)

        # Initialize the adapter(s) for fusion
        adapter_inputs = {
            'clip_dim': 512,
            'mbart_dim': self.encoder_hidden_dim,
            'prefix_length': params.prefix_length,
            'num_heads': params.num_heads,
            'num_layers': params.num_layers,
            'fusion_num_heads': 16,
            'use_gate': params.use_gate,
            'use_fusion': self.use_fusion,
            'mapping_network': params.mapping_network,
        }
        self.adapter = GraphAdapterLarge(**adapter_inputs)
        # if params.mapping_network == 'gat':
        #     self.adapter = GATAdapter(**adapter_inputs)
        # elif params.mapping_network == 'gatl':
        #     self.adapter = GATAdapterLarge(**adapter_inputs)
        # elif params.mapping_network == 'ginel':
        #     self.adapter = GINEAdapterLarge(**adapter_inputs)
        # else:
        #     raise ValueError(
        #         f"Unknown mapping_network: {params.mapping_network}")

        # Use get_lang_code for the target language.
        # (In MBART, typically e.g. "de_DE", "en_XX"; in NLLB, e.g. "deu_Latn", "eng_Latn", etc.)
        self.target_lang = get_lang_code(params.tgt_lang, self.backbone)

        # Decide the generation length
        if 'wmt' in params.test_ds[1] or '2014' in params.test_ds[0]:
            self.max_new_tokens = 300
        else:
            self.max_new_tokens = 60

    def forward(self, batch, mode='train'):
        """
        If mode='train', perform the forward pass and compute loss.
        If mode='test', run generation.
        """
        graph = batch.pop('graph')
        batch.pop('clip')

        # ===========================
        #       TRAIN MODE
        # ===========================
        if mode == 'train':
            # Replace padding tokens (id=1) in labels with -100 so they are ignored in the loss
            batch['mbart']['labels'][batch['mbart']['labels'] == 1] = -100

            # If we use the backbone encoder, run the original encoder, then the adapter
            # Otherwise, only run the adapter
            if self.use_backbone_encoder:
                encoder_outputs = self.backbone_model.model.encoder(
                    input_ids=batch['mbart']['input_ids'],
                    attention_mask=batch['mbart']['attention_mask'],
                    return_dict=True)
                if self.use_fusion:
                    # Adapter does fusion between encoder outputs and the graph
                    encoder_outputs = self.adapter(graph, encoder_outputs)
                else:
                    # Only take the adapter output as final encoder hidden states
                    encoder_outputs.last_hidden_state = self.adapter(graph)
            else:
                # Directly use the adapter output as the encoder outputs
                adapted = self.adapter(graph)
                encoder_outputs = BaseModelOutput(last_hidden_state=adapted)

            # Use the model decoder to compute loss
            outputs = self.backbone_model(
                input_ids=None,  # we pass encoder_outputs instead
                attention_mask=batch['mbart']['attention_mask']
                if self.use_backbone_encoder else None,
                labels=batch['mbart']['labels'],
                encoder_outputs=encoder_outputs,
                return_dict=True)
            return outputs

        # ===========================
        #       TEST MODE
        # ===========================
        elif mode == 'test':
            # Remove labels in inference mode
            batch['mbart'].pop('labels', None)

            if self.use_backbone_encoder:
                encoder_outputs = self.backbone_model.model.encoder(
                    input_ids=batch['mbart']['input_ids'],
                    attention_mask=batch['mbart']['attention_mask'],
                    return_dict=True)
                if self.use_fusion:
                    encoder_outputs = self.adapter(graph, encoder_outputs)
                else:
                    encoder_outputs.last_hidden_state = self.adapter(graph)
            else:
                adapted = self.adapter(graph)
                encoder_outputs = BaseModelOutput(last_hidden_state=adapted)

            # forced_bos_token_id differs in MBART vs. NLLB
            if self.backbone == 'mbart':
                # MBART typically uses lang_code_to_id, e.g. "en_XX", "de_DE"
                forced_bos_id = self.tokenizer.lang_code_to_id[
                    self.target_lang]
            else:  # NLLB typically uses BCP-47 codes, e.g. "eng_Latn", "deu_Latn"
                forced_bos_id = self.tokenizer.convert_tokens_to_ids(
                    self.target_lang)

            # Perform generation
            outputs = self.backbone_model.generate(
                input_ids=None,
                attention_mask=batch['mbart']['attention_mask']
                if self.use_backbone_encoder else None,
                encoder_outputs=encoder_outputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=self.max_new_tokens)
            return outputs


class MBART(nn.Module):

    def __init__(self, params):
        super(MBART, self).__init__()
        if params.image_encoder == 'mclip':
            logger.info('Using MCLIP encoder')
            self.clip = M_CLIP(params)
        elif params.image_encoder == 'clip':
            logger.info('Using CLIP encoder')
            self.clip = CLIP(params)
        elif params.image_encoder == 'clip_res':
            logger.info('Using CLIP-Resnet encoder')
            self.clip = CLIP_RES(params)

        self.backbone = params.backbone  # 'mbart' or 'nllb'

        if self.backbone == 'mbart':
            # MBART-based backbone
            logger.info("Backbone set to mbart-large-50")
            self.tokenizer = MBart50Tokenizer.from_pretrained(
                "facebook/mbart-large-50")
            self.backbone_model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50")
            # MBART-50 has hidden_dim = 1024
            self.encoder_hidden_dim = 1024

        elif self.backbone == 'nllb':
            # NLLB-based backbone
            logger.info(
                "Backbone set to NLLB (facebook/nllb-200-distilled-600M)")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/nllb-200-distilled-600M")
            self.backbone_model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/nllb-200-distilled-600M")
            # The hidden_dim of nllb-200-distilled-600M is also 1024
            self.encoder_hidden_dim = 1024

        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        self.target_lang = get_lang_code(params.tgt_lang, self.backbone)
        logger.info(self.target_lang)

        if 'wmt' in params.test_ds[1] or '2014' in params.test_ds[0]:
            self.max_new_tokens = 300
        else:
            self.max_new_tokens = 60

    def forward(self, batch, mode='train'):
        if mode == 'train':
            labels = batch['mbart']['labels']
            input_ids = batch['mbart']['input_ids']
            attention_mask = batch['mbart']['attention_mask']

            # Replace padding token id's of the labels by -100 so it's ignored by the loss
            labels = labels.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.backbone_model(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          labels=labels,
                                          return_dict=True)
            return outputs

        elif mode == 'test':
            batch['mbart'].pop('labels')
            batch['mbart'].pop('mask_decoder_input_ids')
                        # forced_bos_token_id differs in MBART vs. NLLB
            if self.backbone == 'mbart':
                # MBART typically uses lang_code_to_id, e.g. "en_XX", "de_DE"
                forced_bos_id = self.tokenizer.lang_code_to_id[
                    self.target_lang]
            else:  # NLLB typically uses BCP-47 codes, e.g. "eng_Latn", "deu_Latn"
                forced_bos_id = self.tokenizer.convert_tokens_to_ids(
                    self.target_lang)
            outputs = self.backbone_model.generate(
                **batch['mbart'],
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=self.max_new_tokens)

            return outputs

        else:
            raise ValueError("Mode should be either 'train' or 'test'")


# class MBART(nn.Module):

#     def __init__(self, params):
#         super(MBART, self).__init__()
#         if params.image_encoder == 'mclip':
#             logger.info('Using MCLIP encoder')
#             self.clip = M_CLIP(params)
#         elif params.image_encoder == 'clip':
#             logger.info('Using CLIP encoder')
#             self.clip = CLIP(params)
#         elif params.image_encoder == 'clip_res':
#             logger.info('Using CLIP-Resnet encoder')
#             self.clip = CLIP_RES(params)

#         self.backbone = "mbart"

#         # Initialize tokenizer and mBART model
#         self.tokenizer = MBart50Tokenizer.from_pretrained(
#             'facebook/mbart-large-50')
#         self.mbart = MBartForConditionalGeneration.from_pretrained(
#             'facebook/mbart-large-50')
#         self.target_lang = get_lang_code(
#             params.tgt_lang
#         )  # Assuming get_lang_code is defined in model_utils
#         logger.info(self.target_lang)

#         if 'wmt' in params.test_ds[1] or '2014' in params.test_ds[0]:
#             self.max_new_tokens = 300
#         else:
#             self.max_new_tokens = 60

#     def forward(self, batch, mode='train'):
#         if mode == 'train':
#             labels = batch['mbart']['labels']
#             input_ids = batch['mbart']['input_ids']
#             attention_mask = batch['mbart']['attention_mask']

#             # Replace padding token id's of the labels by -100 so it's ignored by the loss
#             labels = labels.clone()
#             labels[labels == self.tokenizer.pad_token_id] = -100

#             outputs = self.mbart(input_ids=input_ids,
#                                  attention_mask=attention_mask,
#                                  labels=labels,
#                                  return_dict=True)
#             return outputs

#         elif mode == 'test':
#             batch['mbart'].pop('labels')
#             batch['mbart'].pop('mask_decoder_input_ids')
#             outputs = self.mbart.generate(**batch['mbart'],
#                                           forced_bos_token_id=self.tokenizer.
#                                           lang_code_to_id[self.target_lang],
#                                           max_new_tokens=self.max_new_tokens)

#             return outputs

#         else:
#             raise ValueError("Mode should be either 'train' or 'test'")
