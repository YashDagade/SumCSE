import collections
import inspect
import math
import sys
import os
import re
import json
import shutil
import time
import warnings
from pathlib import Path
import importlib.util
from packaging import version
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_torch_tpu_available,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    reissue_pt_warnings,
)

from transformers.utils import logging
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

from transformers.trainer import _model_unwrap
from transformers.optimization import Adafactor, AdamW, get_scheduler
import copy

# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock

logger = logging.get_logger(__name__)


# =========================
# [ANGLES ADDED] Utilities
# =========================
import random
import string
import torch.nn.functional as F
import nltk
from nltk.corpus import brown

nltk.download('brown')

def _generate_cached_random_sentences(n=100, min_length=5, max_length=30) -> List[str]:

    
    sentences = []
    for sentence in brown.sents():
        sentence_str = ' '.join(sentence).strip()
        if not sentence_str.endswith(('.', '!', '?')):
            continue
        if '...' in sentence_str or sentence_str.endswith(',') or sentence_str.endswith(';'):
            continue
        word_count = len(sentence)
        if word_count < min_length or word_count > max_length:
            continue
        sentences.append(sentence_str)
        if len(sentences) >= n:
            break

    return sentences


def _compute_mean_var_angle(embeddings: np.ndarray) -> Tuple[float, float]:
    """
    embeddings: shape (N, dim)
    Return (mean_angle, var_angle) in degrees.
    """
    N = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    dot_products = embeddings @ embeddings.T
    cosines = np.clip(dot_products / (norms * norms.T), -1.0, 1.0)

    angles = []
    for i in range(N):
        for j in range(i+1, N):
            angle_deg = np.degrees(np.arccos(cosines[i,j]))
            angles.append(angle_deg)
    angles = np.array(angles)
    return float(angles.mean()), float(angles.var())  # [CAST FIX ADDED]


class CLTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # [ANGLES ADDED] Create once so we don't generate random sentences at each eval
        self._cached_random_sents = _generate_cached_random_sentences(n=100)


    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            eval_senteval_transfer: bool = False,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch_ = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch_:
                batch_[k] = batch_[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch_, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        self.model.eval()
        results = se.eval(tasks)

        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        metrics = {
            "eval_stsb_spearman": float(stsb_spearman),  # [CAST FIX ADDED]
            "eval_sickr_spearman": float(sickr_spearman),  # [CAST FIX ADDED]
            "eval_avg_sts": float((stsb_spearman + sickr_spearman) / 2.0),
        }
        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0.0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                devacc = float(results[task]['devacc'])  # [CAST FIX ADDED]
                avg_transfer += devacc
                metrics['eval_{}'.format(task)] = devacc
            avg_transfer /= 7.0
            metrics['eval_avg_transfer'] = float(avg_transfer)

        # =========================
        # [ANGLES ADDED] measure angles on cached random sents
        # =========================
        random_embs = self._encode_random_sentences(self._cached_random_sents)
        mean_angle, var_angle = _compute_mean_var_angle(random_embs)
        metrics["mean_angle_bw_embeddings"] = mean_angle
        metrics["variance_bw_embeddings"] = var_angle

        # [CAST FIX ADDED] Convert to float if needed
        # (some of these might still be np.float32, so we cast again)
        for k, v in list(metrics.items()):
            if isinstance(v, (np.float32, np.float64)):
                metrics[k] = float(v)

        # Optionally log LR
        if self.lr_scheduler is not None:
            lrs = self.lr_scheduler.get_last_lr()
            if len(lrs) > 0:
                metrics["learning_rate"] = float(lrs[0])  # [CAST FIX ADDED]

        self.log(metrics)
        return metrics

    def _encode_random_sentences(self, sentences: List[str], batch_size=16) -> np.ndarray:
        device = self.args.device
        self.model.eval()
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sents = sentences[i:i+batch_size]
            inputs = self.tokenizer(
                batch_sents,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=64,
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True)
                # We'll assume outputs.pooler_output is the embedding
                embs = outputs.pooler_output  # shape: (batch_size, hidden_dim)
                embs = F.normalize(embs, p=2, dim=1)
            all_embeddings.append(embs.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()


    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we want to save.
        assert _model_unwrap(model) is self.model, "internal model should be a reference to self.model"

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            import numpy as np
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
            ):
                output_dir = self.args.output_dir
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                self.save_model(output_dir)
                if self.deepspeed:
                    self.deepspeed.save_checkpoint(output_dir)

                if self.sharded_dpp:
                    self.optimizer.consolidate_state_dict()

                if is_torch_tpu_available():
                    import torch_xla.core.xla_model as xm
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        reissue_pt_warnings(caught_warnings)
                elif self.is_world_process_zero() and not self.deepspeed:
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)

                if self.is_world_process_zero():
                    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune
                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
                self.store_flos()

            self.save_model(output_dir)
            if self.deepspeed:
                self.deepspeed.save_checkpoint(output_dir)

            if self.sharded_dpp:
                self.optimizer.consolidate_state_dict()

            if is_torch_tpu_available():
                import torch_xla.core.xla_model as xm
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)
            elif self.is_world_process_zero() and not self.deepspeed:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)

            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True)

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        The main difference between ours and Huggingface's original implementation is that we
        also load model_args when reloading best checkpoints for evaluation.
        """
        self._hp_search_setup(trial)

        if self.model_init is not None:
            set_seed(self.args.seed)
            model = self.call_model_init(trial)
            if not self.is_model_parallel:
                model = model.to(self.args.device)
            self.model = model
            self.model_wrapped = model
            self.optimizer, self.lr_scheduler = None, None

        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        train_dataloader = self.get_train_dataloader()

        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        if self.args.deepspeed:
            model, optimizer, lr_scheduler = init_deepspeed(self, num_training_steps=max_steps)
            self.model = model.module
            self.model_wrapped = model
            self.deepspeed = model
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        else:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self._load_optimizer_and_scheduler(model_path)
        model = self.model_wrapped

        if self.use_apex:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if self.sharded_dpp:
            model = ShardedDDP(model, self.optimizer)
        elif self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )

        if model is not self.model:
            self.model_wrapped = model

        if is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        if train_dataset_is_sized:
            num_examples = self.num_examples(train_dataloader)
        else:
            num_examples = total_train_batch_size * self.args.max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(self.args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            epoch_iterator = train_dataloader

            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(train_dataloader) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            assert train_dataset_is_sized, "currently we only support sized dataloader!"

            inputs = None
            last_inputs = None
            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        steps_in_epoch <= self.args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        if hasattr(self.optimizer, "clip_grad_norm"):
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    if is_torch_tpu_available():
                        import torch_xla.core.xla_model as xm
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    import torch_xla.core.xla_model as xm
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint, model_args=self.model_args)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)
            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)