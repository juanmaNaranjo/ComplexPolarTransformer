import csv
import math
import os
import time
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

from core.metrics import evaluate_regression


def _unwrap_subset(ds):
    """
    Devuelve (dataset_base, indices_en_dataset_base) para Subset anidados.
    Si ds no es Subset, indices=None.
    """
    indices = None
    base = ds

    while isinstance(base, Subset):
        current = list(base.indices)
        if indices is None:
            indices = current
        else:
            indices = [current[i] for i in indices]
        base = base.dataset

    return base, indices


def _as_index_array(indices: Optional[Iterable], n_total: int) -> np.ndarray:
    if indices is None:
        return np.arange(n_total, dtype=np.int64)
    return np.asarray(list(indices), dtype=np.int64)


class Trainer:
    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        test_dl=None,
        lr=1e-3,
        max_epochs=300,
        ckpt_dir="checkpoints",
        log_dir="logs",
        normalize_target=True,
        per_atom_norm=True,
        hparams: dict = None,
        grad_clip=1.0,
        patience=30,
        min_delta=5e-4,
        scheduler_cfg=None,
    ):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.lr = lr
        self.max_epochs = max_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # L1Loss (MAE) como objetivo de entrenamiento. El benchmark QM9 evalúa MAE;
        # entrenar con MSE sesga el modelo hacia reducir outliers a costa del error medio.
        self.loss_fn = torch.nn.L1Loss()

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        self.hparams = hparams or {}
        self.grad_clip = grad_clip
        self.best_val = float("inf")
        self.best_val_mae = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

        self.normalize_target = bool(normalize_target)
        self.per_atom_norm = bool(per_atom_norm)

        y0 = torch.as_tensor(self.train_dl.dataset[0]["y"]).float().view(-1)
        self.num_targets = y0.numel()

        if self.normalize_target:
            self.y_mean, self.y_std = self._compute_target_stats_exact()
        else:
            self.y_mean = torch.zeros(self.num_targets, dtype=torch.float32)
            self.y_std = torch.ones(self.num_targets, dtype=torch.float32)

        print(
            f"[TARGET] normalize_target={self.normalize_target} | "
            f"per_atom_norm={self.per_atom_norm} | "
            f"y_mean={self.y_mean.tolist()} | y_std={self.y_std.tolist()}"
        )

        self.scheduler = None
        if isinstance(scheduler_cfg, dict) and scheduler_cfg.get("name", "") == "reduce_on_plateau":
            factor = float(scheduler_cfg.get("factor", 0.5))
            sched_patience = int(scheduler_cfg.get("patience", 10))
            min_lr = float(scheduler_cfg.get("min_lr", 1e-5))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=factor, patience=sched_patience, min_lr=min_lr
            )

        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)

        self.history = {
            "epoch": [],
            "train_mse": [],
            "val_mse": [],
            "epoch_time_sec": [],
            "train_samples_per_sec": [],
            "val_samples_per_sec": [],
            "peak_gpu_mem_mb": [],
            "lr": [],
        }

        self.csv_path = os.path.join(self.log_dir, "training_log.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch",
                "train_mse",
                "val_mse",
                "epoch_time_sec",
                "train_samples_per_sec",
                "val_samples_per_sec",
                "peak_gpu_mem_mb",
                "lr",
            ])

    def _target_columns(self, base):
        if hasattr(base, "target_cols"):
            return list(getattr(base, "target_cols"))
        if hasattr(base, "target_col"):
            return [getattr(base, "target_col")]
        return None

    def _compute_target_stats_exact(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula media/std usando únicamente train_ds.

        Si per_atom_norm=True, las estadísticas se calculan exactamente sobre:
            y_train_per_atom = y_train / N_atoms
        No se usa aproximación por n_mean=13.
        """
        base, idx = _unwrap_subset(self.train_dl.dataset)
        target_cols = self._target_columns(base)

        if hasattr(base, "df") and target_cols is not None:
            index_array = _as_index_array(idx, len(base))
            vals = base.df.iloc[index_array][target_cols].values.astype("float32")

            if self.per_atom_norm:
                if not hasattr(base, "num_atoms"):
                    raise AttributeError("El dataset no expone num_atoms; no se puede calcular per_atom_norm exacto.")
                n_atoms = np.asarray(base.num_atoms, dtype="float32")[index_array].reshape(-1, 1)
                vals = vals / np.clip(n_atoms, 1.0, None)

            mean = torch.from_numpy(np.mean(vals, axis=0)).float()
            std = torch.from_numpy(np.std(vals, axis=0)).float().clamp_min(1e-9)
            return mean, std

        # Fallback exacto, más lento: recorre el subset de entrenamiento.
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
            return torch.as_tensor(batch["num_atoms"], dtype=torch.float32, device=self.device).view(-1, 1)

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
            y_std = self.y_std.to(self.device)
            y = (y - y_mean) / y_std

        prepared = {
            "atom_types": self._to_device_list(batch["atom_types"]),
            "coords_spherical": self._to_device_list(batch["coords_spherical"]),
            "coords_cart": self._to_device_list(batch.get("coords_cart")),
            "edge_index": self._to_device_list(batch.get("edge_index")),
            "edge_attr": self._to_device_list(batch.get("edge_attr")),
            "y": y,
        }
        if n_atoms is not None:
            prepared["_n_atoms"] = n_atoms
        return prepared

    def _denormalize(self, value_norm, batch_prepared):
        """Desnormaliza predicción/target al espacio original del CSV."""
        value = value_norm
        if self.normalize_target:
            y_std = self.y_std.to(self.device)
            y_mean = self.y_mean.to(self.device)
            value = value * y_std + y_mean

        if self.per_atom_norm and "_n_atoms" in batch_prepared:
            value = value * batch_prepared["_n_atoms"].clamp_min(1.0)
        return value

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        samples_seen = 0

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for batch in self.train_dl:
            batch = self._prepare_batch(batch)
            samples_seen += batch["y"].shape[0]

            pred = self.model(batch)
            pred = torch.as_tensor(pred).float()
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)

            loss = self.loss_fn(pred, batch["y"])
            # set_to_none=True libera tensores de gradiente en lugar de zerofill:
            # reduce ~15-30% la presión de memoria durante el paso de optimización.
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            # Acumular pérdida ponderada por muestra para reportar correctamente
            # cuando el último batch tiene menos muestras que batch_size.
            total_loss += loss.item() * batch["y"].shape[0]

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        return total_loss / max(samples_seen, 1), {
            "train_time_sec": elapsed,
            "train_samples": samples_seen,
            "train_samples_per_sec": samples_seen / max(elapsed, 1e-9),
        }

    def _eval_epoch(self, dl):
        self.model.eval()
        total_loss = 0.0
        preds, targets = [], []
        samples_seen = 0

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in dl:
                batch = self._prepare_batch(batch)
                samples_seen += batch["y"].shape[0]

                pred = self.model(batch)
                pred = torch.as_tensor(pred).float()
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)

                loss = self.loss_fn(pred, batch["y"])
                total_loss += loss.item() * batch["y"].shape[0]

                preds.append(self._denormalize(pred, batch).cpu())
                targets.append(self._denormalize(batch["y"], batch).cpu())

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        metrics = evaluate_regression(preds, targets)

        return total_loss / max(samples_seen, 1), metrics, {
            "time_sec": elapsed,
            "samples": samples_seen,
            "samples_per_sec": samples_seen / max(elapsed, 1e-9),
        }

    def val_epoch(self):
        return self._eval_epoch(self.val_dl)

    def save_ckpt(self, epoch):
        torch.save(
            {
                "model": self.model.state_dict(),
                # optimizer y scheduler se guardan para poder reanudar sin perder
                # momentos de AdamW ni el conteo interno de ReduceLROnPlateau.
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "epoch": epoch,
                "best_val_mae": self.best_val_mae,
                "wait": self.wait,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "normalize_target": self.normalize_target,
                "per_atom_norm": self.per_atom_norm,
                "hparams": self.hparams,
            },
            os.path.join(self.ckpt_dir, "best_model.pt"),
        )

    def fit(self):
        print(
            f"[MODEL] Params: {self.param_count:,} | "
            f"Size: {self.model_size_mb:.2f} MB | "
            f"Device: {self.device} | grad_clip: {self.grad_clip}"
        )

        for epoch in range(1, self.max_epochs + 1):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            epoch_t0 = time.perf_counter()
            train_loss, train_perf = self.train_epoch()
            val_loss, metrics, val_perf = self.val_epoch()
            val_mae_current = metrics.get("mae", float("inf"))

            # Scheduler y early stopping monitorizan val MAE físico (kcal/mol desnorm),
            # no val MSE normalizado. MAE óptimo ≠ MSE óptimo.
            if self.scheduler is not None:
                self.scheduler.step(val_mae_current)

            lr_now = float(self.optimizer.param_groups[0]["lr"])

            if self.device == "cuda":
                torch.cuda.synchronize()
                peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_gpu_mem_mb = None

            epoch_time_sec = time.perf_counter() - epoch_t0
            mae_val = metrics.get("mae", float("nan"))
            r2_val = metrics.get("r2", float("nan"))

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

            # Early stopping sobre val MAE físico (kcal/mol), no sobre la pérdida normalizada.
            if val_mae_current < self.best_val_mae - self.min_delta:
                self.best_val_mae = val_mae_current
                self.best_val = val_loss
                self.wait = 0
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
                    epoch,
                    train_loss,
                    val_loss,
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

        if self.test_dl is not None:
            best_path = os.path.join(self.ckpt_dir, "best_model.pt")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model"])
                self.model.to(self.device)
                self.model.eval()

            test_mse, test_metrics, _ = self._eval_epoch(self.test_dl)
            mae_val = test_metrics.get("mae", float("nan"))
            rmse_val = test_metrics.get("rmse", float("nan"))
            r2_val = test_metrics.get("r2", float("nan"))

            kcal_to_ev = 0.043363
            if math.isfinite(mae_val):
                mae_mev = mae_val * kcal_to_ev * 1000
                print(
                    f"[TEST] MSE: {test_mse:.6f} (norm) | "
                    f"MAE: {mae_val:.4f} kcal/mol | {mae_mev:.2f} meV | "
                    f"RMSE: {rmse_val:.4f} kcal/mol | R2: {r2_val:.6f}"
                )
                print("[TEST] Referencia — SchNet: 0.3130 kcal/mol | MPNN: 0.3550 kcal/mol")
                print(f"[TEST] Factor vs SchNet: {mae_val / 0.3130:.1f}x")
            else:
                print(f"[TEST] MSE: {test_mse:.6f} | MAE: nan | R2: nan")
                print("[TEST WARNING] MAE=nan. Revisa core/metrics.py.")

    def plot(self):
        if not self.history["train_mse"]:
            return
        plt.figure(figsize=(7, 4))
        plt.plot(self.history["train_mse"], label="Train MSE")
        plt.plot(self.history["val_mse"], label="Val MSE")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE normalizado")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png"), dpi=200)
        plt.close()
