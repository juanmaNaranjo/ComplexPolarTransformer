import csv
import math
import os
import time
from typing import Iterable, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # backend sin GUI: no crea objetos Tk → evita crash en DataLoader workers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from core.metrics import evaluate_regression


# ─────────────────────────────────────────────────────────────────────────────
# EMA Helper
# ─────────────────────────────────────────────────────────────────────────────

class EMAHelper:
    """
    Exponential Moving Average de los pesos del modelo.

    Reduce la varianza del modelo final sin costo computacional en inferencia.
    Usado en DimeNet++, SphereNet, PaiNN para mejorar 5-10% el MAE final.

        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> None:
        """Copia pesos EMA al modelo (para evaluación)."""
        self._backup = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
        }
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restaura pesos originales del modelo (tras evaluación)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data.copy_(self._backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay  = state["decay"]
        self.shadow = state["shadow"]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de split
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_subset(ds):
    indices = None
    base = ds
    while isinstance(base, Subset):
        current = list(base.indices)
        indices = current if indices is None else [current[i] for i in indices]
        base = base.dataset
    return base, indices


def _as_index_array(indices: Optional[Iterable], n_total: int) -> np.ndarray:
    if indices is None:
        return np.arange(n_total, dtype=np.int64)
    return np.asarray(list(indices), dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Trainer benchmark para ComplexPolarTransformer v12.

    Mejoras sobre v11 (auditoría científica — invarianza rotacional):
    - EMA de pesos: reduce varianza del modelo final (~5-10% MAE).
    - AMP (Automatic Mixed Precision): ~2× speedup en GPU, batch size mayor.
    - Cosine Annealing con warmup lineal: más estable que ReduceLROnPlateau
      en las primeras epochs y evita mesetas locales.
    - weight_decay expuesto como parámetro (antes hardcodeado en 1e-4).
    - Soporte para ambos schedulers (cosine_with_warmup + reduce_on_plateau).
    - EMA state guardado en checkpoint para reanudar correctamente.
    - phase_reg_weight: minimiza concentración circular de fase para evitar
      colapso (todas las fases convergiendo al mismo valor).
    """

    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        test_dl          = None,
        lr               = 1e-3,
        weight_decay     = 1e-4,
        max_epochs       = 300,
        ckpt_dir         = "checkpoints",
        log_dir          = "logs",
        normalize_target = True,
        per_atom_norm    = True,
        hparams: dict    = None,
        grad_clip        = 1.0,
        patience         = 30,
        min_delta        = 5e-4,
        scheduler_cfg    = None,
        use_ema          = True,
        ema_decay        = 0.999,
        use_amp          = True,
        phase_reg_weight: float = 0.0,
    ):
        self.model    = model
        self.train_dl = train_dl
        self.val_dl   = val_dl
        self.test_dl  = test_dl
        self.lr       = lr
        self.max_epochs = max_epochs
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=float(weight_decay),
        )
        # L1Loss (MAE): benchmark QM9 evalúa MAE; MSE sesgaría hacia outliers.
        self.loss_fn = torch.nn.L1Loss()

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir,  exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.log_dir  = log_dir

        self.hparams      = hparams or {}
        self.grad_clip    = grad_clip
        self.best_val     = float("inf")
        self.best_val_mae = float("inf")
        self.patience     = patience
        self.min_delta    = min_delta
        self.wait         = 0

        self.normalize_target = bool(normalize_target)
        self.per_atom_norm    = bool(per_atom_norm)

        y0 = torch.as_tensor(self.train_dl.dataset[0]["y"]).float().view(-1)
        self.num_targets = y0.numel()

        if self.normalize_target:
            self.y_mean, self.y_std = self._compute_target_stats_exact()
        else:
            self.y_mean = torch.zeros(self.num_targets, dtype=torch.float32)
            self.y_std  = torch.ones(self.num_targets,  dtype=torch.float32)

        print(
            f"[TARGET] normalize_target={self.normalize_target} | "
            f"per_atom_norm={self.per_atom_norm} | "
            f"y_mean={self.y_mean.tolist()} | y_std={self.y_std.tolist()}"
        )

        # ── Scheduler ─────────────────────────────────────────────────────
        self.scheduler = None
        self._scheduler_is_plateau = False

        if isinstance(scheduler_cfg, dict):
            name = scheduler_cfg.get("name", "")

            if name == "reduce_on_plateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=float(scheduler_cfg.get("factor", 0.5)),
                    patience=int(scheduler_cfg.get("patience", 10)),
                    min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
                )
                self._scheduler_is_plateau = True

            elif name == "cosine_with_warmup":
                warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 10))
                min_lr        = float(scheduler_cfg.get("min_lr", 1e-6))
                cosine_epochs = max(self.max_epochs - warmup_epochs, 1)

                warmup_sched  = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_epochs,
                    eta_min=min_lr,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_epochs],
                )
                self._scheduler_is_plateau = False

        # ── EMA ───────────────────────────────────────────────────────────
        self.use_ema = bool(use_ema)
        self.ema     = EMAHelper(self.model, decay=float(ema_decay)) if self.use_ema else None

        # ── AMP ───────────────────────────────────────────────────────────
        # AMP solo activo en CUDA; en CPU float32 es siempre correcto.
        self.use_amp = bool(use_amp) and (self.device == "cuda")
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Regularización de fase ────────────────────────────────────────
        # λ aplicado a la concentración circular R = |mean(e^{iθ})| ∈ [0,1]
        # que el modelo almacena en _phase_reg durante el forward.
        self.phase_reg_weight = float(phase_reg_weight)

        self.param_count  = sum(p.numel() for p in self.model.parameters())
        self.model_size_mb = (
            sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        )

        self.history = {
            "epoch":                [],
            "train_mse":            [],
            "val_mse":              [],
            "epoch_time_sec":       [],
            "train_samples_per_sec": [],
            "val_samples_per_sec":  [],
            "peak_gpu_mem_mb":      [],
            "lr":                   [],
        }

        self.csv_path = os.path.join(self.log_dir, "training_log.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_l1", "val_l1",
                "epoch_time_sec",
                "train_samples_per_sec", "val_samples_per_sec",
                "peak_gpu_mem_mb", "lr",
            ])

    # ── Utilidades de target ──────────────────────────────────────────────

    def _target_columns(self, base):
        if hasattr(base, "target_cols"):
            return list(getattr(base, "target_cols"))
        if hasattr(base, "target_col"):
            return [getattr(base, "target_col")]
        return None

    def _compute_target_stats_exact(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula media/std usando únicamente train_ds.
        Si per_atom_norm=True, estadísticas sobre y_train / N_atoms.
        """
        base, idx = _unwrap_subset(self.train_dl.dataset)
        target_cols = self._target_columns(base)

        if hasattr(base, "df") and target_cols is not None:
            index_array = _as_index_array(idx, len(base))
            vals = base.df.iloc[index_array][target_cols].values.astype("float32")
            if self.per_atom_norm:
                if not hasattr(base, "num_atoms"):
                    raise AttributeError(
                        "El dataset no expone num_atoms; no se puede calcular per_atom_norm exacto."
                    )
                n_atoms = np.asarray(base.num_atoms, dtype="float32")[index_array].reshape(-1, 1)
                vals = vals / np.clip(n_atoms, 1.0, None)
            mean = torch.from_numpy(np.mean(vals, axis=0)).float()
            std  = torch.from_numpy(np.std(vals,  axis=0)).float().clamp_min(1e-9)
            return mean, std

        # Fallback: recorre el subset
        y_list = []
        for i in range(len(self.train_dl.dataset)):
            sample = self.train_dl.dataset[i]
            y = torch.as_tensor(sample["y"]).float().view(-1)
            if self.per_atom_norm:
                n_atoms = float(sample.get("num_atoms", sample["atom_types"].shape[0]))
                y = y / max(n_atoms, 1.0)
            y_list.append(y)
        y_values = torch.stack(y_list, dim=0)
        return y_values.mean(dim=0), y_values.std(dim=0, unbiased=False).clamp_min(1e-9)

    def _batch_n_atoms(self, batch):
        if "num_atoms" in batch and batch["num_atoms"] is not None:
            return torch.as_tensor(
                batch["num_atoms"], dtype=torch.float32, device=self.device
            ).view(-1, 1)
        return torch.tensor(
            [float(at.shape[0]) for at in batch["atom_types"]],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

    def _to_device_list(self, values):
        if values is None:
            return None
        return [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in values]

    def _prepare_batch(self, batch):
        y = torch.as_tensor(batch["y"]).float().to(self.device)
        if y.dim() == 0:
            y = y.view(1, 1)
        elif y.dim() == 1:
            y = y.unsqueeze(-1)

        n_atoms = self._batch_n_atoms(batch) if self.per_atom_norm else None
        if self.per_atom_norm:
            y = y / n_atoms.clamp_min(1.0)
        if self.normalize_target:
            y_mean = self.y_mean.to(self.device)
            y_std  = self.y_std.to(self.device)
            y = (y - y_mean) / y_std

        prepared = {
            "atom_types":      self._to_device_list(batch["atom_types"]),
            "coords_spherical": self._to_device_list(batch["coords_spherical"]),
            "coords_cart":     self._to_device_list(batch.get("coords_cart")),
            "edge_index":      self._to_device_list(batch.get("edge_index")),
            "edge_attr":       self._to_device_list(batch.get("edge_attr")),
            "y":               y,
        }
        if n_atoms is not None:
            prepared["_n_atoms"] = n_atoms
        return prepared

    def _denormalize(self, value_norm, batch_prepared):
        value = value_norm
        if self.normalize_target:
            y_std  = self.y_std.to(self.device)
            y_mean = self.y_mean.to(self.device)
            value  = value * y_std + y_mean
        if self.per_atom_norm and "_n_atoms" in batch_prepared:
            value = value * batch_prepared["_n_atoms"].clamp_min(1.0)
        return value

    # ── Entrenamiento ─────────────────────────────────────────────────────

    def train_epoch(self):
        self.model.train()
        total_loss       = 0.0
        phase_conc_total = 0.0
        phase_conc_count = 0
        samples_seen     = 0

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for batch in self.train_dl:
            batch        = self._prepare_batch(batch)
            batch_size   = batch["y"].shape[0]
            samples_seen += batch_size

            # AMP: autocast solo para ops lineales/conv; trig en float32
            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                pred = self.model(batch)
                pred = torch.as_tensor(pred).float()
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)
                loss = self.loss_fn(pred, batch["y"])
                # _phase_reg = |mean(e^{iθ})| ∈ [0,1]; lo calcula el modelo en forward.
                # Minimizarlo mantiene la distribución de fases diversa (no colapsada).
                phase_reg = getattr(self.model, "_phase_reg", None)
                if phase_reg is not None and self.phase_reg_weight > 0:
                    loss = loss + self.phase_reg_weight * phase_reg.float()

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.grad_clip:
                # unscale antes de clip para que la norma sea real
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Actualizar EMA tras cada step de optimizer
            if self.use_ema and self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item() * batch_size

            # Acumula concentración de fase para monitoreo (sin gradiente)
            if phase_reg is not None:
                phase_conc_total += phase_reg.detach().item() * batch_size
                phase_conc_count += batch_size

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return total_loss / max(samples_seen, 1), {
            "train_time_sec":        elapsed,
            "train_samples":         samples_seen,
            "train_samples_per_sec": samples_seen / max(elapsed, 1e-9),
            "phase_concentration":   phase_conc_total / max(phase_conc_count, 1),
        }

    def _eval_epoch(self, dl):
        """Evalúa con pesos EMA si están disponibles."""
        if self.use_ema and self.ema is not None:
            self.ema.apply(self.model)

        self.model.eval()
        total_loss   = 0.0
        preds, targets = [], []
        samples_seen = 0

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in dl:
                batch      = self._prepare_batch(batch)
                batch_size = batch["y"].shape[0]
                samples_seen += batch_size

                with torch.autocast(device_type=self.device, enabled=self.use_amp):
                    pred = self.model(batch)
                pred = torch.as_tensor(pred).float()
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)

                loss = self.loss_fn(pred, batch["y"])
                total_loss += loss.item() * batch_size

                preds.append(self._denormalize(pred, batch).cpu())
                targets.append(self._denormalize(batch["y"], batch).cpu())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if self.use_ema and self.ema is not None:
            self.ema.restore(self.model)

        preds   = torch.cat(preds,   dim=0)
        targets = torch.cat(targets, dim=0)
        metrics = evaluate_regression(preds, targets)

        return total_loss / max(samples_seen, 1), metrics, {
            "time_sec":        elapsed,
            "samples":         samples_seen,
            "samples_per_sec": samples_seen / max(elapsed, 1e-9),
        }

    def val_epoch(self):
        return self._eval_epoch(self.val_dl)

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save_ckpt(self, epoch):
        ckpt = {
            "model":           self.model.state_dict(),
            "optimizer":       self.optimizer.state_dict(),
            "scheduler":       self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler":          self.scaler.state_dict(),
            "epoch":           epoch,
            "best_val_mae":    self.best_val_mae,
            "wait":            self.wait,
            "y_mean":          self.y_mean,
            "y_std":           self.y_std,
            "normalize_target": self.normalize_target,
            "per_atom_norm":   self.per_atom_norm,
            "hparams":         self.hparams,
        }
        if self.use_ema and self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()
        torch.save(ckpt, os.path.join(self.ckpt_dir, "best_model.pt"))

    # ── Ciclo principal ───────────────────────────────────────────────────

    def fit(self):
        print(
            f"[MODEL] Params: {self.param_count:,} | "
            f"Size: {self.model_size_mb:.2f} MB | "
            f"Device: {self.device} | AMP: {self.use_amp} | "
            f"EMA: {self.use_ema} | grad_clip: {self.grad_clip}"
        )

        for epoch in range(1, self.max_epochs + 1):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            epoch_t0 = time.perf_counter()
            train_loss, train_perf = self.train_epoch()
            val_loss, metrics, val_perf = self.val_epoch()
            val_mae_current = metrics.get("mae", float("inf"))

            # Scheduler: cosine_with_warmup → step() sin args
            #            reduce_on_plateau  → step(metric)
            if self.scheduler is not None:
                if self._scheduler_is_plateau:
                    self.scheduler.step(val_mae_current)
                else:
                    self.scheduler.step()

            lr_now = float(self.optimizer.param_groups[0]["lr"])

            if self.device == "cuda":
                torch.cuda.synchronize()
                peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_gpu_mem_mb = None

            epoch_time_sec = time.perf_counter() - epoch_t0
            mae_val = metrics.get("mae", float("nan"))
            r2_val  = metrics.get("r2",  float("nan"))

            msg = (
                f"Epoch {epoch}/{self.max_epochs} | "
                f"Train L1 {train_loss:.4f} | Val L1 {val_loss:.4f} | "
                f"Val MAE {mae_val:.4f} | Val R2 {r2_val:.6f} | "
                f"LR {lr_now:.2e} | Time {epoch_time_sec:.2f}s | "
                f"Train {train_perf['train_samples_per_sec']:.0f} s/s | "
                f"Val {val_perf['samples_per_sec']:.0f} s/s"
            )
            if peak_gpu_mem_mb is not None:
                msg += f" | GPU {peak_gpu_mem_mb:.1f} MB"
            print(msg)

            # Monitoreo de diversidad de fase cada 10 epochs
            phase_conc = train_perf.get("phase_concentration", float("nan"))
            if epoch % 10 == 0 and not math.isnan(phase_conc) and phase_conc > 0.0:
                status = "COLAPSO" if phase_conc > 0.9 else ("WARN" if phase_conc > 0.5 else "OK")
                print(
                    f"  [FASE] Concentración circular = {phase_conc:.4f} [{status}] "
                    f"(0=diversa, 1=colapso)"
                )

            # Early stopping sobre val MAE físico (kcal/mol desnorm)
            if val_mae_current < self.best_val_mae - self.min_delta:
                self.best_val_mae = val_mae_current
                self.best_val     = val_loss
                self.wait         = 0
                self.save_ckpt(epoch)
            else:
                self.wait += 1

            self.history["epoch"].append(epoch)
            self.history["train_mse"].append(train_loss)
            self.history["val_mse"].append(val_loss)
            self.history["epoch_time_sec"].append(epoch_time_sec)
            self.history["train_samples_per_sec"].append(train_perf["train_samples_per_sec"])
            self.history["val_samples_per_sec"].append(val_perf["samples_per_sec"])
            self.history["peak_gpu_mem_mb"].append(peak_gpu_mem_mb or 0.0)
            self.history["lr"].append(lr_now)

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    epoch, train_loss, val_loss,
                    epoch_time_sec,
                    train_perf["train_samples_per_sec"],
                    val_perf["samples_per_sec"],
                    peak_gpu_mem_mb or "",
                    lr_now,
                ])

            if self.wait >= self.patience:
                print(f"[EARLY STOPPING] Mejor Val MAE: {self.best_val_mae:.6f} kcal/mol")
                break

        self.plot()

        # ── Evaluación final en test con el mejor checkpoint ──────────────
        if self.test_dl is not None:
            best_path = os.path.join(self.ckpt_dir, "best_model.pt")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model"])
                # Restaurar EMA si existe en el checkpoint
                if self.use_ema and self.ema is not None and "ema" in ckpt:
                    self.ema.load_state_dict(ckpt["ema"])
                self.model.to(self.device)
                self.model.eval()

            test_loss, test_metrics, _ = self._eval_epoch(self.test_dl)
            mae_val  = test_metrics.get("mae",  float("nan"))
            rmse_val = test_metrics.get("rmse", float("nan"))
            r2_val   = test_metrics.get("r2",   float("nan"))

            kcal_to_ev = 0.043363
            if math.isfinite(mae_val):
                mae_mev = mae_val * kcal_to_ev * 1000
                print(
                    f"[TEST] L1: {test_loss:.6f} (norm) | "
                    f"MAE: {mae_val:.4f} kcal/mol | {mae_mev:.2f} meV | "
                    f"RMSE: {rmse_val:.4f} kcal/mol | R2: {r2_val:.6f}"
                )
                print("[TEST] Referencia — SchNet: 0.3130 kcal/mol | MPNN: 0.3550 kcal/mol")
                print(f"[TEST] Factor vs SchNet: {mae_val / 0.3130:.1f}×")
            else:
                print(f"[TEST] L1: {test_loss:.6f} | MAE: nan | R2: nan")
                print("[TEST WARNING] MAE=nan. Revisa core/metrics.py.")

    def plot(self):
        if not self.history["train_mse"]:
            return
        plt.figure(figsize=(7, 4))
        plt.plot(self.history["train_mse"], label="Train L1")
        plt.plot(self.history["val_mse"],   label="Val L1")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("L1 (normalizado)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png"), dpi=200)
        plt.close()
