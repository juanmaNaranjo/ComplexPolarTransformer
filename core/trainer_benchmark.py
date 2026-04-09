import os
import csv
import time
import torch
import matplotlib.pyplot as plt
from core.metrics import evaluate_regression


def _unwrap_subset(ds):
    """
    Devuelve (base_dataset, indices) si ds es Subset (posiblemente anidado).
    Si no es Subset, retorna (ds, None)
    """
    indices = None
    base = ds
    # evitar import circular en runtime si torch no tiene Subset aquí:
    from torch.utils.data import Subset

    while isinstance(base, Subset):
        indices = base.indices if indices is None else [base.indices[i] for i in indices]
        base = base.dataset
    return base, indices


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
        hparams: dict = None,
        grad_clip=5.0,
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
        self.loss_fn = torch.nn.MSELoss()

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir

        self.hparams = hparams or {}
        self.grad_clip = grad_clip

        # ============================
        # Early stopping
        # ============================
        self.best_val = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0

        # ============================
        # Normalización del target (rápida)
        # ============================
        self.normalize_target = normalize_target

        # Inferir num_targets del primer item (single: 1)
        y0 = torch.as_tensor(self.train_dl.dataset[0]["y"]).float().view(-1)
        self.num_targets = y0.numel()

        if self.normalize_target:
            self.y_mean, self.y_std = self._compute_target_stats_fast()
        else:
            self.y_mean = torch.zeros(self.num_targets, dtype=torch.float32)
            self.y_std = torch.ones(self.num_targets, dtype=torch.float32)

        # ============================
        # Scheduler (ReduceLROnPlateau)
        # ============================
        self.scheduler = None
        if isinstance(scheduler_cfg, dict) and scheduler_cfg.get("name", "") == "reduce_on_plateau":
            factor = float(scheduler_cfg.get("factor", 0.5))
            patience = int(scheduler_cfg.get("patience", 10))
            min_lr = float(scheduler_cfg.get("min_lr", 1e-5))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
            )

        # ============================
        # Métricas de eficiencia del modelo
        # ============================
        self.param_count = sum(p.numel() for p in self.model.parameters())
        self.model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)

        # ============================
        # Historial
        # ============================
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

        # CSV
        self.csv_path = os.path.join(self.log_dir, "training_log.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_mse",
                "val_mse",
                "epoch_time_sec",
                "train_samples_per_sec",
                "val_samples_per_sec",
                "peak_gpu_mem_mb",
                "lr",
            ])

    def _compute_target_stats_fast(self):
        """
        Intenta calcular mean/std desde el DataFrame del dataset (rápido).
        Si no existe df, hace fallback a un muestreo.
        """
        base, idx = _unwrap_subset(self.train_dl.dataset)
        target_cols = None

        # Soportar tanto target_cols como target_col
        if hasattr(base, "target_cols"):
            target_cols = list(getattr(base, "target_cols"))
        elif hasattr(base, "target_col"):
            target_cols = [getattr(base, "target_col")]
        else:
            target_cols = None

        if hasattr(base, "df") and target_cols is not None:
            import numpy as np
            df = getattr(base, "df")
            if idx is None:
                vals = df[target_cols].values.astype("float32")
            else:
                vals = df.iloc[idx][target_cols].values.astype("float32")
            # vals shape: (N, T)
            mean = torch.from_numpy(np.mean(vals, axis=0)).float()
            std = torch.from_numpy(np.std(vals, axis=0)).float().clamp_min(1e-9)
            return mean, std

        # Fallback: muestreo (evita iterar 110k items con RDKit)
        sample_n = min(5000, len(self.train_dl.dataset))
        y_list = []
        for i in range(sample_n):
            yi = torch.as_tensor(self.train_dl.dataset[i]["y"]).float().view(-1)
            y_list.append(yi)
        y_values = torch.stack(y_list, dim=0)
        mean = y_values.mean(dim=0)
        std = y_values.std(dim=0, unbiased=False).clamp_min(1e-9)
        return mean, std

    def _prepare_batch(self, batch):
        y = torch.as_tensor(batch["y"]).float().to(self.device)
        if y.dim() == 0:
            y = y.view(1, 1)
        elif y.dim() == 1:
            y = y.unsqueeze(-1)

        if self.normalize_target:
            y_mean = self.y_mean.to(self.device)
            y_std = self.y_std.to(self.device)
            y = (y - y_mean) / y_std

        return {
            "atom_types": [x.to(self.device) for x in batch["atom_types"]],
            "coords_spherical": [c.to(self.device) for c in batch["coords_spherical"]],
            "coords_cart": [cc.to(self.device) for cc in batch["coords_cart"]],
            "edge_index": [ei.to(self.device) for ei in batch["edge_index"]],
            "edge_attr": [ea.to(self.device) for ea in batch["edge_attr"]],
            "y": y,
        }

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

            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        throughput = samples_seen / max(elapsed, 1e-9)

        return total_loss / len(self.train_dl), {
            "train_time_sec": elapsed,
            "train_samples": samples_seen,
            "train_samples_per_sec": throughput,
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
                total_loss += loss.item()

                # des-normalizar para métricas en unidad original
                p = pred * self.y_std.to(self.device) + self.y_mean.to(self.device)
                t = batch["y"] * self.y_std.to(self.device) + self.y_mean.to(self.device)

                preds.append(p.cpu())
                targets.append(t.cpu())

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        throughput = samples_seen / max(elapsed, 1e-9)

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = evaluate_regression(preds, targets)
        return total_loss / len(dl), metrics, {
            "time_sec": elapsed,
            "samples": samples_seen,
            "samples_per_sec": throughput,
        }

    def val_epoch(self):
        return self._eval_epoch(self.val_dl)

    def save_ckpt(self, epoch):
        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "hparams": self.hparams,
            },
            os.path.join(self.ckpt_dir, "best_model.pt"),
        )

    def fit(self):
        print(
            f"[MODEL] Params: {self.param_count:,} | "
            f"Size: {self.model_size_mb:.2f} MB | "
            f"Device: {self.device}"
        )

        for epoch in range(1, self.max_epochs + 1):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            epoch_t0 = time.perf_counter()

            train_mse, train_perf = self.train_epoch()
            val_mse, metrics, val_perf = self.val_epoch()

            # scheduler step (ReduceLROnPlateau)
            if self.scheduler is not None:
                self.scheduler.step(val_mse)

            lr_now = float(self.optimizer.param_groups[0]["lr"])

            if self.device == "cuda":
                torch.cuda.synchronize()
                peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_gpu_mem_mb = None

            epoch_time_sec = time.perf_counter() - epoch_t0

            msg = (
                f"Epoch {epoch}/{self.max_epochs} | "
                f"Train MSE {train_mse:.4f} | "
                f"Val MSE {val_mse:.4f} | "
                f"LR {lr_now:.2e} | "
                f"Epoch Time {epoch_time_sec:.2f}s | "
                f"Train Throughput {train_perf['train_samples_per_sec']:.2f} samples/s | "
                f"Val Throughput {val_perf['samples_per_sec']:.2f} samples/s"
            )
            if peak_gpu_mem_mb is not None:
                msg += f" | Peak GPU Mem {peak_gpu_mem_mb:.2f} MB"
            print(msg)

            # Early stopping
            if val_mse < self.best_val - self.min_delta:
                self.best_val = val_mse
                self.wait = 0
                self.save_ckpt(epoch)
            else:
                self.wait += 1

            self.history["epoch"].append(epoch)
            self.history["train_mse"].append(train_mse)
            self.history["val_mse"].append(val_mse)
            self.history["epoch_time_sec"].append(epoch_time_sec)
            self.history["train_samples_per_sec"].append(train_perf["train_samples_per_sec"])
            self.history["val_samples_per_sec"].append(val_perf["samples_per_sec"])
            self.history["peak_gpu_mem_mb"].append(peak_gpu_mem_mb if peak_gpu_mem_mb is not None else 0.0)
            self.history["lr"].append(lr_now)

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch,
                    train_mse,
                    val_mse,
                    epoch_time_sec,
                    train_perf["train_samples_per_sec"],
                    val_perf["samples_per_sec"],
                    peak_gpu_mem_mb if peak_gpu_mem_mb is not None else "",
                    lr_now,
                ])

            if self.wait >= self.patience:
                print(f"[EARLY STOPPING] Mejor Val MSE: {self.best_val:.4f}")
                break

        self.plot()

        # Evaluación final en TEST con el mejor modelo
        if self.test_dl is not None:
            best_path = os.path.join(self.ckpt_dir, "best_model.pt")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model"])
                self.model.to(self.device)
                self.model.eval()

            test_mse, test_metrics, _ = self._eval_epoch(self.test_dl)
            mae_val  = test_metrics.get("mae",  float("nan"))
            rmse_val = test_metrics.get("rmse", float("nan"))
            r2_val   = test_metrics.get("r2",   float("nan"))

            # NOTA: u0_atom en el CSV de QM9 está en kcal/mol.
            # El MAE ya está desnormalizado a kcal/mol — no requiere conversión.
            # Conversión adicional: 1 kcal/mol = 0.04336 eV = 43.36 meV
            import math
            KCAL_TO_EV  = 0.043363
            EV_TO_MEV   = 1000.0
            if math.isfinite(mae_val):
                mae_ev  = mae_val * KCAL_TO_EV
                mae_mev = mae_ev  * EV_TO_MEV
                print(
                    f"[TEST] MSE: {test_mse:.6f} (norm) | "
                    f"MAE: {mae_val:.4f} kcal/mol | "
                    f"{mae_mev:.2f} meV | "
                    f"RMSE: {rmse_val:.4f} kcal/mol | "
                    f"R2: {r2_val:.6f}"
                )
                # Referencia benchmark QM9 u0_atom
                print(f"[TEST] Referencia — SchNet: 0.3130 kcal/mol | MPNN: 0.3550 | NequIP: 0.0420")
                print(f"[TEST] Factor vs SchNet: {mae_val/0.3130:.1f}x")
            else:
                print(f"[TEST] MSE: {test_mse:.6f} | MAE: nan | R2: nan")
                print("[TEST WARNING] MAE=nan. Revisa core/metrics.py — puede haber NaN en predicciones o targets.")

    def plot(self):
        plt.plot(self.history["train_mse"], label="Train MSE")
        plt.plot(self.history["val_mse"], label="Val MSE")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png"))
        plt.close()