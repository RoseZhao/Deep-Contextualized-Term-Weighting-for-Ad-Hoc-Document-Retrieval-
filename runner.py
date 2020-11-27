import logging
import math
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers.data.data_collator import DataCollator, default_data_collator
from transformers.data.datasets import GlueDataTrainingArguments as DataTrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from transformers.training_args import TrainingArguments, is_torch_tpu_available
from utils import weighted_mse_loss
import pdb

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_args: DataTrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None
    read_from_cache: Optional[bool] = False
    num_train_examples: Optional[int] = None

    def __init__(
            self,
            model: PreTrainedModel,
            args: TrainingArguments,
            data_args: DataTrainingArguments,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            prediction_loss_only=False,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
            num_train_examples: Optional[int] = None,
            tokenizer_name: Optional[str] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        self.data_args = data_args
        self.data_collator = data_collator if data_collator is not None else default_data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.tokenizer_name = tokenizer_name

        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)

    def get_train_dataloader(self, train_dataset) -> DataLoader:
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )
        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )
        return data_loader

    def get_optimizers(
            self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def num_iterations(self, dataloader: DataLoader) -> int:
        if self.num_train_examples is None and dataloader is None:
            raise ValueError("Both num_train_examples and dataloader are None!")
        batch_size = self.args.train_batch_size
        return len(dataloader) if dataloader is not None else \
            (self.num_train_examples + batch_size - 1) // batch_size


    def save_ckpt(self, model, optimizer, scheduler, epoch=None):
        # In all cases (even distributed/parallel), self.model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            assert model.module is self.model
        else:
            assert model is self.model
        # Save model checkpoint
        output_dir = \
            os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}") if epoch is None \
                else os.path.join(self.args.output_dir, f"epoch{epoch}-{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

        self.save_model(output_dir)

        if self.is_world_master():
            self._rotate_checkpoints()
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def train_on_dataloader(self, model, optimizer, scheduler, train_dataloader, epoch, steps_trained_in_current_epoch,
                             tr_loss, logging_loss):
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())
        for step, inputs in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            tr_loss += self._training_step(model, inputs, optimizer)

            if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                self.global_step += 1
                self.epoch = epoch + (step + 1) / len(epoch_iterator)

                if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                ):
                    logs: Dict[str, float] = {}
                    logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                    # backward compatibility for pytorch schedulers
                    logs["learning_rate"] = (
                        scheduler.get_last_lr()[0]
                        if version.parse(torch.__version__) >= version.parse("1.4")
                        else scheduler.get_lr()[0]
                    )
                    logging_loss = tr_loss

                    self._log(logs)

                if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                    self.save_ckpt(model, optimizer, scheduler)

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                epoch_iterator.close()
                break

        return steps_trained_in_current_epoch, tr_loss, logging_loss


    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = None if self.train_dataset is None else self.get_train_dataloader(self.train_dataset)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (self.args.max_steps // (self.num_iterations(train_dataloader) //
                                                        self.args.gradient_accumulation_steps) + 1)
        else:
            t_total = int(self.num_iterations(train_dataloader) // self.args.gradient_accumulation_steps *
                          self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train!
        total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (self.num_iterations(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                        self.num_iterations(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if not self.read_from_cache:
                steps_trained_in_current_epoch, tr_loss, logging_loss = self.train_on_dataloader(
                    model, optimizer, scheduler, train_dataloader, epoch, steps_trained_in_current_epoch, tr_loss,
                    logging_loss
                )
            else:   # TODO: modify this
                cache_files = self.get_cache_files("train", self.tokenizer_name)
                random.shuffle(cache_files)
                for cache_file in cache_files:
                    logger.info("training on cache_file %s" % cache_file)
                    train_dataloader, dataset = self.get_dataloader_from_cache_file(cache_file)
                    steps_trained_in_current_epoch, tr_loss, logging_loss = self.train_on_dataloader(
                        model, optimizer, scheduler, train_dataloader, epoch, steps_trained_in_current_epoch, tr_loss,
                        logging_loss
                    )
                    del train_dataloader
                    del dataset
            self.save_ckpt(model, optimizer, scheduler, epoch=epoch)

            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break

        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(self, dataloader: DataLoader, description: str) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader))
        logger.info("  Batch size = %d", batch_size)
        preds: torch.Tensor = None
        token_ids: torch.Tensor = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)
            with torch.no_grad():
                logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                               token_type_ids=inputs["token_type_ids"])
                # logging.info(f"logits = {logits}")
                # logging.info(f"logits.shape = {logits.shape}")
                # logging.info(f"target_mask = {inputs['target_mask']}")
                # logging.info(f"target_mask.shape = {inputs['target_mask'].shape}")

                logits = logits.squeeze(2)
                logits = F.relu(logits)     # zeros out negative weights

            if preds is None:
                preds = logits.detach()
            else:
                preds = torch.cat((preds, logits.detach()), dim=0)

            if token_ids is None:
                token_ids = inputs["input_ids"].detach()
            else:
                token_ids = torch.cat((token_ids, inputs["input_ids"].detach()), dim=0)


        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if token_ids is not None:
                token_ids = self.distributed_concat(token_ids, num_total_examples=self.num_examples(dataloader))

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if token_ids is not None:
            token_ids = token_ids.cpu().numpy()

        return preds, token_ids

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)
            with open(os.path.join(self.args.output_dir, "loss.out"), "a") as f:
                f.write(str(output))
                f.write("\n")

    def _training_step(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():     # TODO: only move the ones that BertModel needs?
            inputs[k] = v.to(self.args.device)
        # input_ids, attention_mask, token_type_ids = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
        # logging.info(f"input_ids = {input_ids}, {input_ids.shape}")
        # logging.info(f"attention_mask = {attention_mask}, {attention_mask.shape}")
        # logging.info(f"token_type_ids = {token_type_ids}, {token_type_ids.shape}")
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"])
        loss = weighted_mse_loss(logits, inputs["target_weights"], inputs["target_mask"])

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        return loss.item()

    def is_local_master(self) -> bool:
        return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """
        if self.is_world_master():
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return
        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output
