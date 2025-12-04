import os
import torch
from tqdm import tqdm
import sacrebleu
import torch.optim as optim
from torch_poly_lr_decay import PolynomialLRDecay
from ddp import *
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import math
import logging
import warnings
import evaluate
import glob
from typing import List, Tuple
from model_utils import get_lang_code, send_to_cuda, EarlyStopMonitor

logger = logging.getLogger("__main__")


def get_model_module(model):
    """Retrieve the underlying model from a DDP-wrapped model."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def get_unfrozen_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield param


class Runner:

    def __init__(self, train_dl, test_dl, params):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.update_count = params.update_count
        self.test_after = params.test_after * self.update_count
        self.is_pretraining = params.stage in [
            'caption', 'text_recon', 'distill'
        ]
        self.enable_layerwise_unfreezing = params.enable_layerwise_unfreezing
        self.unfreeze_epoch_interval = params.unfreeze_epoch_interval
        self.layers_per_unfreeze_epoch_interval = params.layers_per_unfreeze_epoch_interval
        self.current_unfrozen_layers = 0  # Track the number of unfrozen layers
        self.meteor = evaluate.load('meteor')
        self.early_stopper = EarlyStopMonitor(max_round=params.patience,
                                              higher_better=True,
                                              tolerance=1e-5)
        self.best_bleu_test = -float('Inf')

        # Placeholder for model_module; will be set in train method
        self.model_module = None

    def freeze_decoder_layers(self):
        model_module = self.model_module
        self.total_decoder_layers = len(
            model_module.backbone_model.model.decoder.layers)

        for layer in model_module.backbone_model.model.decoder.layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.current_unfrozen_layers = 0
        logger.info(
            f"Initially frozen all {self.total_decoder_layers} decoder layers."
        )

    def unfreeze_next_decoder_layer(self):
        if self.current_unfrozen_layers >= self.total_decoder_layers:
            return

        layer_to_unfreeze = self.total_decoder_layers - self.current_unfrozen_layers - 1
        model_module = self.model_module
        layer = model_module.backbone_model.model.decoder.layers[layer_to_unfreeze]
        for param in layer.parameters():
            param.requires_grad = True
        self.current_unfrozen_layers += 1
        logger.info(
            f"Unfroze decoder layer {layer_to_unfreeze + 1}/{self.total_decoder_layers}."
        )

    def update_optimizer_with_new_layers(self, params):

        model_module = self.model_module
        existing_params = set(p for group in self.optimizer.param_groups
                              for p in group['params'])

        new_layers_params = []
        for layer in model_module.backbone_model.model.decoder.layers[
                -self.current_unfrozen_layers:]:
            for param in layer.parameters():
                if param.requires_grad and (param not in existing_params):
                    new_layers_params.append(param)

        if new_layers_params:

            current_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.add_param_group({
                'params': new_layers_params,
                'lr': current_lr
            })
            logger.info(
                f"Added {len(new_layers_params)} parameters to the optimizer.")
        else:
            logger.info("No new parameters to add to the optimizer.")

    def save_model(self, model, name, epoch):
        model_module = get_model_module(model)  # Use the helper function
        if self.is_pretraining:
            # Stage 1：save adapter and decoder
            checkpoint = {
                'adapter': model_module.adapter.state_dict(),
                'decoder': model_module.backbone_model.model.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict(),
                'epoch': epoch,
                'best_bleu_test': self.best_bleu_test
            }
        else:
            # Stage 2：save whole model
            checkpoint = {
                'model': model_module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scaler': self.scaler.state_dict(),
                'epoch': epoch,
                'best_bleu_test': self.best_bleu_test
            }
        torch.save(checkpoint, name)
        logger.info(f"Model saved to {name} at epoch {epoch}.")

    def load_model(self, params, model, name, load_opt):
        model_module = get_model_module(model)  # Use the helper function
        if not os.path.exists(name) and os.path.exists(
                name.replace('.pth', '_1.pth')):
            model_names = sorted(glob.glob(name.replace('.pth', '*')))
            checkpoint = {
                'model':
                torch.load(model_names[0], map_location=torch.device('cpu')),
                'epoch':
                0,
                'best_bleu_test':
                0
            }
            for model_name in model_names[1:]:
                checkpoint['model'].update(
                    torch.load(model_name, map_location=torch.device('cpu')))
        else:
            checkpoint = torch.load(name, map_location=torch.device('cpu'))

        if self.is_pretraining:
            # Stage 1：load adapter and decoder
            adapter_state = checkpoint.get('adapter', {})
            decoder_state = checkpoint.get('decoder', {})
            model_module.adapter.load_state_dict(adapter_state, strict=False)
            model_module.backbone_model.model.decoder.load_state_dict(decoder_state,
                                                             strict=False)
            logger.info(f"Loaded adapter and decoder from {name}.")
        else:
            # Stage 2：load whole model
            model_state = checkpoint.get('model', {})
            if params.num_gpus == 1:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state.items():
                    name_key = k[7:] if 'module' in k[:7] else k
                    new_state_dict[name_key] = v
                model_state = new_state_dict
            model_module.load_state_dict(model_state, strict=False)
            logger.info(f"Loaded full model from {name}.")

        if load_opt:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'],
                                               strict=False)
                self.scaler.load_state_dict(checkpoint['scaler'], strict=False)
                logger.info("Loaded optimizer and scaler states.")
            except Exception as e:
                warnings.warn(f'Could not load optimizer due to: {e}')
        else:
            warnings.warn(
                'Not loading optimizer - if you intended to continue training (and not just load weights), set load_opt = True and rerun'
            )
        model.cuda()

        return checkpoint.get('epoch',
                              0), checkpoint.get('best_bleu_test',
                                                 -float('Inf'))

    def fit_one_epoch(self, model, tokenizer, params, epoch):
        model.train()
        train_loss = 0.0
        self.optimizer.zero_grad()
        model_module = get_model_module(model)  # Access the underlying model
        batch_loss = 0.0  # Initialize batch loss for TensorBoard logging

        for step, batch in enumerate(
                tqdm(self.train_dl,
                     desc=f'Epoch {epoch}',
                     disable=not is_main_process())):
            batch['mbart'], batch['clip'] = send_to_cuda(
                batch['mbart']), send_to_cuda(batch['clip'])
            batch['graph'] = batch['graph'].cuda()
            with torch.autocast(device_type='cuda'):
                output = model(batch)
                loss = output[0] / self.update_count
            self.scaler.scale(loss).backward()
            batch_loss += loss.item()  # Accumulate loss

            if (step + 1) % self.update_count == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.cycle_scheduler.step()
                self.optimizer.zero_grad()
                batch_loss = 0.0

            if params.num_gpus > 1:
                reduced_loss = loss.detach().clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                train_loss += reduced_loss.item()
            else:
                train_loss += loss.item()
            del batch
            if self.test_after > 0 and (step + 1) % self.test_after == 0:
                # self.test(model, tokenizer, params, epoch)
                self.test_with_analysis(model, tokenizer, params, epoch)

        if is_main_process():
            avg_train_loss = train_loss / len(self.train_dl)
            logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss}\n')
            if self.is_pretraining:
                self.save_model(model,
                                f'{params.model_name}/model_pretrained.pth',
                                epoch)
            else:
                self.save_model(
                    model, f'{params.model_name}/model_epoch_{epoch}.pth',
                    epoch)

    def test(self, model, tokenizer, params, epoch):
        model.eval()
        translated_sentences, target_sentences = [], []
        tokenizer.tgt_lang = get_lang_code(params.tgt_lang, params.backbone)
        model_module = get_model_module(model)  # Access the underlying model

        with torch.no_grad():
            for i, batch in enumerate(
                    tqdm(self.test_dl,
                         desc=f'Epoch {epoch}',
                         disable=not is_main_process())):
                batch['clip'] = send_to_cuda(batch['clip'])
                batch['mbart'] = send_to_cuda(batch['mbart'])
                batch['graph'] = batch['graph'].cuda()
                raw_target_text = batch.pop('raw')
                with torch.autocast(device_type='cuda'):
                    output = model(batch, mode='test')
                output = tokenizer.batch_decode(output,
                                                skip_special_tokens=True)
                if params.num_gpus > 1:
                    output_list = [None for _ in range(params.num_gpus)]
                    dist.all_gather_object(output_list, output)
                    output = []
                    for out in output_list:
                        output.extend(out)

                    target_list = [None for _ in range(params.num_gpus)]
                    dist.all_gather_object(target_list, raw_target_text)
                    raw_target_text = []
                    for tgt in target_list:
                        raw_target_text.extend(tgt)
                else:
                    output = output
                    raw_target_text = raw_target_text

                translated_sentences.extend(output)
                target_sentences.extend(raw_target_text)

        if is_main_process():
            bleu_score = sacrebleu.corpus_bleu(translated_sentences,
                                               [target_sentences]).score
            meteor_score = self.meteor.compute(
                predictions=translated_sentences,
                references=target_sentences)['meteor']
            logger.info(
                f'Epoch {epoch}; Test BLEU: {bleu_score}; Test METEOR: {100 * meteor_score}'
            )
            logger.info('------------------------------------------')
            for i, (tra, tgt) in enumerate(
                    zip(translated_sentences[0:5], target_sentences[0:5])):
                logger.info(f'Target Sentence {i}: {tgt}')
                logger.info(f'Translated Sentence {i}: {tra}')
                logger.info('------------------------------------------')

            if bleu_score > self.best_bleu_test and not self.is_pretraining and not params.test:
                self.best_bleu_test = bleu_score
                self.save_model(model,
                                f'{params.model_name}/model_best_test.pth',
                                epoch)
            return bleu_score
        else:
            return None

    def test_with_analysis(self, model, tokenizer, params, epoch, top_k=1000):

        model.eval()
        translated_sentences = []
        target_sentences = []
        source_sentences = []
        tokenizer.tgt_lang = get_lang_code(params.tgt_lang, params.backbone)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_dl, desc=f'Epoch {epoch}', disable=not is_main_process())):

                source_ids = batch['mbart']['input_ids']
                tokenizer.src_lang = get_lang_code(params.src_lang, params.backbone)
                source_text = tokenizer.batch_decode(source_ids, skip_special_tokens=True)
                tokenizer.tgt_lang = get_lang_code(params.tgt_lang, params.backbone)

                batch['clip'] = send_to_cuda(batch['clip'])
                batch['mbart'] = send_to_cuda(batch['mbart'])
                batch['graph'] = batch['graph'].cuda()
                raw_target_text = batch.pop('raw')

                with torch.autocast(device_type='cuda'):
                    output = model(batch, mode='test')
                output = tokenizer.batch_decode(output, skip_special_tokens=True)

                if params.num_gpus > 1:
                    output_list = [None for _ in range(params.num_gpus)]
                    dist.all_gather_object(output_list, output)
                    output = []
                    for out in output_list:
                        output.extend(out)

                    target_list = [None for _ in range(params.num_gpus)]
                    dist.all_gather_object(target_list, raw_target_text)
                    raw_target_text = []
                    for tgt in target_list:
                        raw_target_text.extend(tgt)

                    source_list = [None for _ in range(params.num_gpus)]
                    dist.all_gather_object(source_list, source_text)
                    source_text = []
                    for src in source_list:
                        source_text.extend(src)

                translated_sentences.extend(output)
                target_sentences.extend(raw_target_text)
                source_sentences.extend(source_text)

        if is_main_process():

            bleu_score = sacrebleu.corpus_bleu(translated_sentences, [target_sentences]).score
            meteor_score = self.meteor.compute(predictions=translated_sentences, references=target_sentences)['meteor']

            logger.info(f'Epoch {epoch}; Test BLEU: {bleu_score}; Test METEOR: {100 * meteor_score}')
            logger.info('='*80)
            logger.info('Translation Analysis (Sorted by sentence BLEU score)')
            logger.info('='*80)

            sentence_scores = []
            for src, tgt, trans in zip(source_sentences, target_sentences, translated_sentences):
                score = sacrebleu.sentence_bleu(trans, [tgt]).score
                sentence_scores.append((score, src, tgt, trans))

            sentence_scores.sort(reverse=True)

            for i, (score, src, tgt, trans) in enumerate(sentence_scores[:top_k], 1):
                logger.info(f"\nExample #{i} - BLEU Score: {score:.2f}")
                logger.info(f"{params.src_lang:<10}: {src}")
                logger.info(f"{params.tgt_lang:<10}: {tgt}")
                logger.info(f"{'Trans':<10}: {trans}")
                logger.info("-"*80)

            if bleu_score > self.best_bleu_test and not self.is_pretraining and not params.test:
                self.best_bleu_test = bleu_score
                self.save_model(model, f'{params.model_name}/model_best_test.pth', epoch)

            return bleu_score
        else:
            return None

    def train(self, model, tokenizer, params):
        self.model_module = get_model_module(model)

        if self.is_pretraining:
            weight_decay = 0.01
        else:
            weight_decay = 0.0

        initial_params = list(get_unfrozen_params(self.model_module))

        self.optimizer = optim.AdamW(initial_params,
                                     lr=params.lr,
                                     betas=(0.9, 0.98),
                                     eps=1e-6,
                                     weight_decay=weight_decay)
        logger.info(
            "Initialized AdamW optimizer with currently unfrozen parameters only."
        )

        self.scaler = GradScaler()
        self.cycle_scheduler = PolynomialLRDecay(self.optimizer,
                                                 max_decay_steps=40000,
                                                 end_learning_rate=params.lr /
                                                 2,
                                                 power=2.0)

        if params.load_model:
            last_epoch, self.best_bleu_test = self.load_model(
                params,
                model,
                f'{params.model_name}/{params.load_model}'
                if '/' not in params.load_model else
                f'{params.model_dir}/{params.load_model}',
                load_opt=False)
            steps_per_epoch = math.ceil(len(self.train_dl) / self.update_count)
            if params.continue_training:
                last_batch = (last_epoch - 1) * steps_per_epoch
                for step in range(last_batch):
                    self.cycle_scheduler.step()
                logger.info(f"Resumed training from epoch {last_epoch}.")
            else:
                last_epoch = 0
        else:
            last_epoch = 0

        if params.test:
            # self.test(model, tokenizer, params, last_epoch)
            self.test_with_analysis(model, tokenizer, params, last_epoch)
            return

        for epoch in range(last_epoch, params.epochs):
            if params.num_gpus > 1:
                self.train_dl.sampler.set_epoch(epoch)
                self.test_dl.sampler.set_epoch(epoch)
            self.fit_one_epoch(model, tokenizer, params, epoch + 1)
            bleu_score = self.test(model, tokenizer, params, epoch + 1)

            if is_main_process():
                if self.is_pretraining and self.enable_layerwise_unfreezing:
                    if self.current_unfrozen_layers >= self.total_decoder_layers:
                        should_stop = self.early_stopper.early_stop_check(
                            bleu_score)
                    else:
                        should_stop = False
                else:
                    should_stop = self.early_stopper.early_stop_check(
                        bleu_score)
            else:
                should_stop = None

            if params.num_gpus > 1:
                if should_stop is None:
                    should_stop_tensor = torch.tensor(0).cuda()
                else:
                    should_stop_tensor = torch.tensor(
                        1 if should_stop else 0).cuda()
                dist.broadcast(should_stop_tensor, src=0)
                should_stop = bool(should_stop_tensor.item())

            if should_stop:
                if is_main_process():
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break


def compute_sentence_bleu(hypothesis: str, reference: str) -> float:

    return sacrebleu.sentence_bleu(hypothesis, [reference]).score


def analyze_translations(
        source_texts: List[str],
        target_texts: List[str],
        translated_texts: List[str],
        top_k: int = None) -> List[Tuple[str, str, str, float]]:


    assert len(source_texts) == len(target_texts) == len(translated_texts)

    results = []
    for src, tgt, trans in zip(source_texts, target_texts, translated_texts):
        bleu_score = compute_sentence_bleu(trans, tgt)
        results.append((src, tgt, trans, bleu_score))


    results.sort(key=lambda x: x[3], reverse=True)

    if top_k is not None:
        results = results[:top_k]

    return results


def print_analysis_results(results: List[Tuple[str, str, str, float]],
                           source_lang: str = "Source",
                           target_lang: str = "Target"):
    logger.info("\n" + "=" * 80)
    logger.info("Translation Analysis Results (Sorted by BLEU Score)")
    logger.info("=" * 80)

    for i, (src, tgt, trans, bleu) in enumerate(results, 1):
        logger.info(f"\nExample #{i} - BLEU Score: {bleu:.2f}")
        logger.info(f"{source_lang:<10}: {src}")
        logger.info(f"{target_lang:<10}: {tgt}")
        logger.info(f"{'Trans':<10}: {trans}")
        logger.info("-" * 80)
