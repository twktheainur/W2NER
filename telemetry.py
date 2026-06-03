"""
Training telemetry for W2NER investigation.

Provides instrumentation to understand why W2NER outperforms ADAPTBioEL:
- Per-batch/epoch timing breakdown (data loading → forward → backward)
- GPU memory tracking (peak, allocated, reserved)
- Gradient norm tracking per parameter group (BERT vs non-BERT)
- Per-zone loss decomposition (NNW vs THW vs PAD)
- Learning rate tracking per step
- TensorBoard logging for visualization

Design: dependency-injected into Trainer. Stateless accumulator that resets
per-epoch. Follows functional patterns for data collection.
"""

import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


def create_tensorboard_writer(log_dir):
    """Create a TensorBoard SummaryWriter.

    Returns None if tensorboard is not installed (graceful fallback).
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        return writer
    except (ImportError, ModuleNotFoundError):
        return None


class TrainingTelemetry:
    """Collects and reports training telemetry data per epoch.

    Usage:
        telemetry = TrainingTelemetry(config, logger, writer)
        for epoch in range(epochs):
            telemetry.start_epoch(epoch)
            for batch in loader:
                telemetry.start_batch()
                data = load()
                telemetry.record_data_loaded()
                output = forward()
                telemetry.record_forward_done()
                loss.backward()
                telemetry.record_backward_done()
                telemetry.record_loss(loss.item())
                telemetry.record_gpu_memory()
            telemetry.log_epoch_summary(train_f1, eval_f1, best_f1, best_test_f1)
    """

    def __init__(self, config, logger, writer=None):
        self.config = config
        self.logger = logger
        self.writer = writer

        # Timing accumulators (seconds, reset per-epoch)
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.eval_time = 0.0

        # Counters
        self.batch_count = 0
        self._epoch = 0

        # Per-batch tracking lists (reset per-epoch)
        self.losses = []
        self.zone_losses = {"nnw": [], "thw": [], "pad": []}
        self.learning_rates = defaultdict(list)
        self.grad_norms = defaultdict(list)

        # GPU memory (reset per-epoch)
        self.gpu_memory = {
            "allocated_mb": [],
            "max_allocated_mb": [],
            "reserved_mb": [],
        }

        # Internal timers
        self._epoch_start = None
        self._last_tick = None

    # ── Epoch lifecycle ────────────────────────────────────────────

    def start_epoch(self, epoch):
        """Reset all accumulators for a new epoch."""
        self._epoch = epoch
        self._epoch_start = time.perf_counter()
        self._last_tick = self._epoch_start
        self._reset_accumulators()

    # ── Batch lifecycle ────────────────────────────────────────────

    def start_batch(self):
        """Mark the start of a new batch (before data loading)."""
        self._last_tick = time.perf_counter()

    def record_data_loaded(self):
        """Time spent loading + transferring data to GPU."""
        if self._last_tick is not None:
            self.data_time += time.perf_counter() - self._last_tick
        self._last_tick = time.perf_counter()

    def record_forward_done(self):
        """Time spent in forward pass (model + loss computation)."""
        if self._last_tick is not None:
            self.forward_time += time.perf_counter() - self._last_tick
        self._last_tick = time.perf_counter()

    def record_backward_done(self):
        """Time spent in backward pass + optimizer step + scheduler."""
        if self._last_tick is not None:
            self.backward_time += time.perf_counter() - self._last_tick
        self._last_tick = time.perf_counter()
        self.batch_count += 1

    # ── Data recording ─────────────────────────────────────────────

    def record_loss(self, loss_val):
        """Record total loss value for the current batch."""
        self.losses.append(loss_val)

    def record_zone_losses(self, outputs, grid_labels, grid_mask2d):
        """Decompose CrossEntropyLoss into per-zone contributions.

        Zones:
          PAD (0):  no relation (background cell)
          NNW (1):  next-neighboring-word relation
          THW (2+): tail-head-word relation (entity type ID)

        Computed on detached outputs — no gradient flow.
        """
        flat_outputs = outputs[grid_mask2d].detach()
        flat_labels = grid_labels[grid_mask2d]

        for zone_name, condition in [
            ("pad", flat_labels == 0),
            ("nnw", flat_labels == 1),
            ("thw", flat_labels >= 2),
        ]:
            mask = condition
            if mask.any():
                zone_loss = nn.functional.cross_entropy(
                    flat_outputs[mask], flat_labels[mask]
                )
                self.zone_losses[zone_name].append(zone_loss.item())

    def record_grad_norms(self, model):
        """Record gradient L2 norm per parameter, plus group aggregates.

        Groups parameters into BERT vs non-BERT and records:
        - Per-parameter gradient norm (for histogram)
        - Mean gradient norm per group (for scalar tracking)
        - BERT/non-BERT gradient ratio (diagnostic)
        """
        bert_norms = []
        non_bert_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                self.grad_norms[name].append(norm)
                if "bert" in name:
                    bert_norms.append(norm)
                else:
                    non_bert_norms.append(norm)

        # Group aggregates
        if bert_norms:
            self.grad_norms["_group_bert"].append(float(np.mean(bert_norms)))
        if non_bert_norms:
            self.grad_norms["_group_non_bert"].append(float(np.mean(non_bert_norms)))
        if bert_norms and non_bert_norms:
            self.grad_norms["_group_ratio"].append(
                float(np.mean(bert_norms) / (np.mean(non_bert_norms) + 1e-12))
            )

    def record_lr(self, optimizer):
        """Record current learning rate for each parameter group."""
        for i, group in enumerate(optimizer.param_groups):
            self.learning_rates[i].append(group["lr"])

    def record_gpu_memory(self):
        """Record current GPU memory allocation stats."""
        if not torch.cuda.is_available():
            return
        self.gpu_memory["allocated_mb"].append(
            torch.cuda.memory_allocated() / 1024**2
        )
        self.gpu_memory["max_allocated_mb"].append(
            torch.cuda.max_memory_allocated() / 1024**2
        )
        try:
            self.gpu_memory["reserved_mb"].append(
                torch.cuda.memory_reserved() / 1024**2
            )
        except AttributeError:
            pass  # Older PyTorch

    # ── Logging ────────────────────────────────────────────────────

    def log_batch_metrics(self, loss_val, lr_val):
        """Return compact metrics dict for tqdm / progress bar."""
        metrics = {"loss": f"{loss_val:.4f}", "lr": f"{lr_val:.2e}"}
        if self.gpu_memory["max_allocated_mb"]:
            peak = self.gpu_memory["max_allocated_mb"][-1]
            metrics["gpu_mb"] = f"{peak:.0f}"
        return metrics

    def log_epoch_summary(self, train_f1, eval_f1, best_f1, best_test_f1):
        """Log structured epoch summary to logger and TensorBoard."""
        epoch_elapsed = time.perf_counter() - self._epoch_start

        avg_loss = float(np.mean(self.losses)) if self.losses else 0.0
        avg_nnw = float(np.mean(self.zone_losses["nnw"])) if self.zone_losses["nnw"] else 0.0
        avg_thw = float(np.mean(self.zone_losses["thw"])) if self.zone_losses["thw"] else 0.0
        avg_pad = float(np.mean(self.zone_losses["pad"])) if self.zone_losses["pad"] else 0.0

        # ── Build readable summary ──
        lines = [
            f"Epoch {self._epoch} Telemetry",
            f"  Duration: {epoch_elapsed:.1f}s | Batches: {self.batch_count} | "
            f"Speed: {self.batch_count / (epoch_elapsed + 1e-12):.1f} batch/s",
        ]

        # Loss breakdown
        lines.append(
            f"  Loss: Total={avg_loss:.4f} | NNW={avg_nnw:.4f} | "
            f"THW={avg_thw:.4f} | PAD={avg_pad:.4f}"
        )

        # F1 summary
        lines.append(
            f"  F1 -> Train: {train_f1:.4f} | Dev: {eval_f1:.4f} | "
            f"Best Dev: {best_f1:.4f} | Best Test: {best_test_f1:.4f}"
        )

        # Timing breakdown
        total = self.forward_time + self.backward_time + self.data_time
        if total > 0:
            pct_fwd = 100.0 * self.forward_time / total
            pct_bwd = 100.0 * self.backward_time / total
            pct_data = 100.0 * self.data_time / total
            lines.append(
                f"  Time: Fwd={self.forward_time:.1f}s ({pct_fwd:.0f}%) | "
                f"Bwd={self.backward_time:.1f}s ({pct_bwd:.0f}%) | "
                f"Data={self.data_time:.1f}s ({pct_data:.0f}%)"
            )

        # GPU memory
        if self.gpu_memory["max_allocated_mb"]:
            peak = max(self.gpu_memory["max_allocated_mb"])
            lines.append(f"  GPU Memory: {peak:.0f} MB peak")

        # Gradient norms
        for key, label in [
            ("_group_non_bert", "Grad Norm (non-BERT)"),
            ("_group_bert", "Grad Norm (BERT)"),
            ("_group_ratio", "Grad Ratio (BERT/non-BERT)"),
        ]:
            if key in self.grad_norms and self.grad_norms[key]:
                val = float(np.mean(self.grad_norms[key]))
                lines.append(f"  {label}: {val:.6f}")

        summary = "\n".join(lines)
        self.logger.info(f"\n{summary}")

        # ── TensorBoard logging ──
        if self.writer is not None:
            self._write_tensorboard(
                epoch=self._epoch,
                avg_loss=avg_loss,
                avg_nnw=avg_nnw,
                avg_thw=avg_thw,
                avg_pad=avg_pad,
                epoch_elapsed=epoch_elapsed,
                train_f1=train_f1,
                eval_f1=eval_f1,
                best_f1=best_f1,
            )

        return summary

    # ── Private helpers ────────────────────────────────────────────

    def _write_tensorboard(self, epoch, avg_loss, avg_nnw, avg_thw, avg_pad,
                           epoch_elapsed, train_f1, eval_f1, best_f1):
        """Write all epoch-level metrics to TensorBoard."""
        writer = self.writer

        # Loss
        writer.add_scalar("loss/total", avg_loss, epoch)
        writer.add_scalar("loss/nnw", avg_nnw, epoch)
        writer.add_scalar("loss/thw", avg_thw, epoch)
        writer.add_scalar("loss/pad", avg_pad, epoch)

        # F1 scores
        writer.add_scalar("f1/train", train_f1, epoch)
        writer.add_scalar("f1/dev", eval_f1, epoch)
        writer.add_scalar("f1/best_dev", best_f1, epoch)

        # Timing
        writer.add_scalar("timing/epoch_seconds", epoch_elapsed, epoch)
        writer.add_scalar("timing/forward_seconds", self.forward_time, epoch)
        writer.add_scalar("timing/backward_seconds", self.backward_time, epoch)
        writer.add_scalar("timing/data_seconds", self.data_time, epoch)
        writer.add_scalar(
            "timing/batches_per_second",
            self.batch_count / (epoch_elapsed + 1e-12),
            epoch,
        )

        # GPU memory
        if self.gpu_memory["max_allocated_mb"]:
            writer.add_scalar(
                "gpu/peak_memory_mb",
                max(self.gpu_memory["max_allocated_mb"]),
                epoch,
            )

        # Gradient norms (histograms for individual params, scalars for groups)
        for name, norms in self.grad_norms.items():
            if name.startswith("_"):
                # Group aggregate → scalar
                if norms:
                    writer.add_scalar(
                        f"gradients/{name}", float(np.mean(norms)), epoch
                    )
            else:
                # Individual param → histogram (sample if many steps)
                if norms and len(norms) > 3:
                    writer.add_histogram(
                        f"gradients/{name}", np.array(norms), epoch
                    )

        # Zone loss histograms
        for zone, losses in self.zone_losses.items():
            if losses and len(losses) > 1:
                writer.add_histogram(
                    f"zone_losses/{zone}", np.array(losses), epoch
                )

        # Learning rates
        for group_id, lrs in self.learning_rates.items():
            if lrs:
                writer.add_scalar(
                    f"learning_rate/group_{group_id}",
                    float(np.mean(lrs)),
                    epoch,
                )

        writer.flush()

    def _reset_accumulators(self):
        """Zero out all per-epoch accumulators."""
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.eval_time = 0.0
        self.batch_count = 0
        self.losses = []
        self.zone_losses = {"nnw": [], "thw": [], "pad": []}
        self.learning_rates = defaultdict(list)
        self.grad_norms = defaultdict(list)
        self.gpu_memory = {
            "allocated_mb": [],
            "max_allocated_mb": [],
            "reserved_mb": [],
        }
