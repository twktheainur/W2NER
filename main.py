"""
W2NER: Unified Named Entity Recognition as Word-Word Relation Classification

Training entry point with telemetry, AMP, early stopping, and reproducibility.
"""

import argparse
import json
import random
import time

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config
import data_loader
import utils
from model import Model
from telemetry import TrainingTelemetry, create_tensorboard_writer

# Optional tqdm progress bar
try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    _tqdm = None


class Trainer:
    """W2NER Trainer with telemetry, AMP, and gradient accumulation support."""

    def __init__(self, model, config, updates_total, telemetry=None, writer=None):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.telemetry = telemetry
        self.writer = writer

        # ── Optimizer with BERT-specific learning rates ──
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [
                    p for n, p in model.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": config.bert_learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.bert.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": config.bert_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(config.warm_factor * updates_total),
            num_training_steps=updates_total,
        )

        # ── AMP scaler (no-op when amp=False) ──
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    # ── Training ────────────────────────────────────────────────────

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        # Reset gradients for accumulation
        self.optimizer.zero_grad()

        if self.telemetry:
            self.telemetry.start_epoch(epoch)

        # Progress bar setup
        if HAS_TQDM and self.config.telemetry:
            iterator = _tqdm(data_loader, desc=f"Epoch {epoch}", unit="batch", leave=False)
        else:
            iterator = enumerate(data_loader)

        for i, data_batch in enumerate(iterator):
            if self.telemetry:
                self.telemetry.start_batch()

            # Move data to GPU
            data_batch = [data.cuda() for data in data_batch[:-1]]
            (
                bert_inputs,
                grid_labels,
                grid_mask2d,
                pieces2word,
                dist_inputs,
                sent_length,
            ) = data_batch

            if self.telemetry:
                self.telemetry.record_data_loaded()

            # ── Forward pass (optionally with AMP) ──
            with torch.cuda.amp.autocast(enabled=self.config.amp):
                outputs = model(
                    bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length
                )
                grid_mask2d = grid_mask2d.clone()
                loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
                # Normalize loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation

            if self.telemetry:
                self.telemetry.record_forward_done()
                self.telemetry.record_loss(
                    loss.item() * self.config.gradient_accumulation
                )
                # Per-zone loss decomposition
                self.telemetry.record_zone_losses(
                    outputs, grid_labels, grid_mask2d
                )
                self.telemetry.record_gpu_memory()

            # ── Backward pass (scaled for AMP) ──
            self.scaler.scale(loss).backward()

            if self.telemetry:
                self.telemetry.record_backward_done()

            # ── Gradient accumulation: step every N batches ──
            if (i + 1) % self.config.gradient_accumulation == 0:
                self._optimizer_step()

            # ── Collect predictions for F1 computation ──
            loss_list.append(loss.cpu().item())

            with torch.no_grad():
                outputs = torch.argmax(outputs, -1)
                _grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                _outputs = outputs[grid_mask2d].contiguous().view(-1)
                label_result.append(_grid_labels.cpu())
                pred_result.append(_outputs.cpu())

            # Update progress bar
            if self.telemetry and HAS_TQDM:
                iterator.set_postfix(
                    self.telemetry.log_batch_metrics(
                        loss.item() * self.config.gradient_accumulation,
                        self.optimizer.param_groups[0]["lr"],
                    )
                )

        # Handle remaining gradients from incomplete accumulation
        if len(data_loader) % self.config.gradient_accumulation != 0:
            self._optimizer_step()

        # ── Compute epoch metrics ──
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(), pred_result.numpy(), average="macro"
        )

        table = pt.PrettyTable([f"Train {epoch}", "Loss", "F1", "Precision", "Recall"])
        table.add_row(
            ["Label", f"{np.mean(loss_list):.4f}"] +
            [f"{x:3.4f}" for x in [f1, p, r]]
        )
        config.logger.info(f"\n{table}")

        # TensorBoard per-epoch train metrics
        if self.writer is not None:
            self.writer.add_scalar("train/epoch_loss", float(np.mean(loss_list)), epoch)
            self.writer.add_scalar("train/epoch_f1", f1, epoch)

        return f1

    # ── Evaluation ──────────────────────────────────────────────────

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        eval_start = time.perf_counter()

        with torch.no_grad():
            for _, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                (
                    bert_inputs,
                    grid_labels,
                    grid_mask2d,
                    pieces2word,
                    dist_inputs,
                    sent_length,
                ) = data_batch

                outputs = model(
                    bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length
                )
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(
                    outputs.cpu().numpy(), entity_text, length.cpu().numpy()
                )

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        eval_time = time.perf_counter() - eval_start

        if self.telemetry:
            self.telemetry.eval_time += eval_time

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(), pred_result.numpy(), average="macro"
        )
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        config.logger.info(
            f"{title} Label F1 {f1_score(label_result.numpy(), pred_result.numpy(), average=None)}"
        )

        table = pt.PrettyTable([f"{title} {epoch}", "F1", "Precision", "Recall"])
        table.add_row(["Label"] + [f"{x:3.4f}" for x in [f1, p, r]])
        table.add_row(["Entity"] + [f"{x:3.4f}" for x in [e_f1, e_p, e_r]])
        config.logger.info(f"\n{table}")

        # TensorBoard eval/test metrics
        if self.writer is not None:
            prefix = "eval/" if not is_test else "test/"
            self.writer.add_scalar(f"{prefix}entity_f1", e_f1, epoch)
            self.writer.add_scalar(f"{prefix}label_f1", f1, epoch)

        return e_f1

    # ── Inference ───────────────────────────────────────────────────

    def predict(self, epoch, data_loader, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i : i + self.config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                (
                    bert_inputs,
                    grid_labels,
                    grid_mask2d,
                    pieces2word,
                    dist_inputs,
                    sent_length,
                ) = data_batch

                outputs = model(
                    bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length
                )
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(
                    outputs.cpu().numpy(), entity_text, length.cpu().numpy()
                )

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append(
                            {
                                "text": [sentence[x] for x in ent[0]],
                                "type": config.vocab.id_to_label(ent[1]),
                            }
                        )
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += self.config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(), pred_result.numpy(), average="macro"
        )
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        config.logger.info(
            f"{title} Label F1 {f1_score(label_result.numpy(), pred_result.numpy(), average=None)}"
        )

        table = pt.PrettyTable([f"{title} {epoch}", "F1", "Precision", "Recall"])
        table.add_row(["Label"] + [f"{x:3.4f}" for x in [f1, p, r]])
        table.add_row(["Entity"] + [f"{x:3.4f}" for x in [e_f1, e_p, e_r]])
        config.logger.info(f"\n{table}")

        with open(self.config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1

    # ── Checkpointing ───────────────────────────────────────────────

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    # ── Private ─────────────────────────────────────────────────────

    def _optimizer_step(self):
        """Unscale, clip, step, and zero gradients (with AMP support)."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.clip_grad_norm
        )

        if self.telemetry:
            self.telemetry.record_grad_norms(self.model)
            self.telemetry.record_lr(self.optimizer)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/conll03.json")
    parser.add_argument("--save_path", type=str, default="./model.pt")
    parser.add_argument("--predict_path", type=str, default="./output.json")
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dist_emb_size", type=int)
    parser.add_argument("--type_emb_size", type=int)
    parser.add_argument("--lstm_hid_size", type=int)
    parser.add_argument("--conv_hid_size", type=int)
    parser.add_argument("--bert_hid_size", type=int)
    parser.add_argument("--ffnn_hid_size", type=int)
    parser.add_argument("--biaffine_size", type=int)
    parser.add_argument("--dilation", type=str, help="e.g. 1,2,3")

    parser.add_argument("--emb_dropout", type=float)
    parser.add_argument("--conv_dropout", type=float)
    parser.add_argument("--out_dropout", type=float)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--clip_grad_norm", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)

    parser.add_argument("--bert_name", type=str)
    parser.add_argument("--bert_learning_rate", type=float)
    parser.add_argument("--warm_factor", type=float)
    parser.add_argument("--use_bert_last_4_layers", type=int, help="1: true, 0: false")

    parser.add_argument("--seed", type=int)

    # ── Telemetry and training improvements ──
    parser.add_argument("--telemetry", type=int, help="1: enable telemetry, 0: disable")
    parser.add_argument("--telemetry_dir", type=str, help="telemetry output directory")
    parser.add_argument("--tensorboard", type=int, help="1: enable TensorBoard, 0: disable")
    parser.add_argument("--amp", type=int, help="1: enable AMP, 0: disable")
    parser.add_argument("--early_stop_patience", type=int, help="early stopping patience (0=disabled)")
    parser.add_argument("--gradient_accumulation", type=int, help="gradient accumulation steps")

    args = parser.parse_args()

    # ── Load config ──
    config = config.Config(args)

    # ── Setup logging ──
    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    # ── Device setup ──
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # ── Reproducibility ──
    if config.seed is not None and config.seed > 0:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info(f"Reproducibility: seed={config.seed}, deterministic=True")

    # ── TensorBoard writer ──
    writer = None
    if config.tensorboard:
        tb_dir = f"{config.telemetry_dir}/tensorboard/{config.dataset}"
        writer = create_tensorboard_writer(tb_dir)
        if writer is not None:
            logger.info(f"TensorBoard logging to {tb_dir}")
        config.logger = logger

    # ── Telemetry ──
    telemetry = None
    if config.telemetry:
        telemetry = TrainingTelemetry(config, logger, writer)
        logger.info(f"Training telemetry enabled (dir: {config.telemetry_dir})")

    # ── Load data ──
    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=data_loader.collate_fn,
            shuffle=i == 0,
            num_workers=4,
            drop_last=i == 0,
        )
        for i, dataset in enumerate(datasets)
    )

    updates_total = (
        len(datasets[0]) // config.batch_size * config.epochs
    )
    if config.gradient_accumulation > 1:
        updates_total = updates_total // config.gradient_accumulation
        logger.info(
            f"Gradient accumulation: {config.gradient_accumulation} steps "
            f"({updates_total} effective updates)"
        )

    # ── Build model ──
    logger.info("Building Model")
    model = Model(config)
    model = model.cuda()

    trainer = Trainer(model, config, updates_total, telemetry=telemetry, writer=writer)

    # ── Training loop ──
    best_f1 = 0.0
    best_test_f1 = 0.0
    patience_counter = 0

    for i in range(config.epochs):
        epoch_start = time.perf_counter()
        logger.info(f"Epoch: {i}")

        train_f1 = trainer.train(i, train_loader)
        dev_f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)

        epoch_time = time.perf_counter() - epoch_start

        # ── Log telemetry epoch summary ──
        if telemetry:
            telemetry.log_epoch_summary(
                train_f1=train_f1,
                eval_f1=dev_f1,
                best_f1=best_f1,
                best_test_f1=best_test_f1,
            )

        # ── Checkpoint if best dev F1 ──
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_test_f1 = test_f1
            trainer.save(config.save_path)
            logger.info(
                f"New best model: Dev F1={best_f1:.4f}, Test F1={best_test_f1:.4f}"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"No improvement for {patience_counter} epoch(s) (best Dev F1={best_f1:.4f})"
            )

        # ── Early stopping ──
        if (
            config.early_stop_patience > 0
            and patience_counter >= config.early_stop_patience
        ):
            logger.info(
                f"Early stopping triggered after {i + 1} epochs "
                f"(patience={config.early_stop_patience})"
            )
            break

    # ── Final results ──
    logger.info(f"Best DEV F1: {best_f1:.4f}")
    logger.info(f"Best TEST F1: {best_test_f1:.4f}")

    # ── Final prediction on test set ──
    trainer.load(config.save_path)
    trainer.predict("Final", test_loader, ori_data[-1])

    # ── Cleanup ──
    if writer is not None:
        writer.close()
