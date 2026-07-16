"""
run_baseline_profiler.py — Captura de línea base completa (v12)

Instrumenta por separado:
  - Tiempo de carga de datos (DataLoader → GPU transfer)
  - Tiempo de forward pass
  - Tiempo de backward pass
  - Throughput train y val
  - Pico de memoria GPU por época
  - MAE, RMSE, R² desnormalizados tras cada época

Produce:
  logs/baseline_metrics.json  — datos crudos por época y batch
  logs/BASELINE.md            — resumen legible

NO modifica ningún archivo del proyecto.
Ejecutar con el mismo entorno que main_train_benchmark.py:
    python run_baseline_profiler.py --config experiments/beta_train_benchmark.yaml --epochs 5
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from core.dataset import QM9SDFDataset
from core.metrics import evaluate_regression
from core.trainer_benchmark import EMAHelper, _unwrap_subset, _as_index_array
from core.utils import set_seed
from models.complex_model_beta import ComplexPolarTransformerBeta


# ─── Helpers ─────────────────────────────────────────────────────────────────

def collate_mol(batch):
    return {
        "coords_cart":     [item["coords_cart"]     for item in batch],
        "coords_spherical":[item["coords_spherical"] for item in batch],
        "atom_types":      [item["atom_types"]       for item in batch],
        "edge_index":      [item["edge_index"]       for item in batch],
        "edge_attr":       [item["edge_attr"]        for item in batch],
        "y":               torch.stack([item["y"] for item in batch]),
        "num_atoms":       torch.tensor([item["num_atoms"] for item in batch], dtype=torch.float32),
        "num_edges":       torch.tensor([item["num_edges"] for item in batch], dtype=torch.long),
        "original_idx":    torch.tensor([item.get("original_idx", -1) for item in batch], dtype=torch.long),
    }


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_mol,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def to_device_list(values, device):
    if values is None:
        return None
    return [v.to(device) if isinstance(v, torch.Tensor) else v for v in values]


def prepare_batch(batch, device, y_mean, y_std, normalize, per_atom):
    y = torch.as_tensor(batch["y"]).float().to(device)
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    n_atoms = batch["num_atoms"].float().to(device).view(-1, 1)
    if per_atom:
        y = y / n_atoms.clamp_min(1.0)
    if normalize:
        y = (y - y_mean.to(device)) / y_std.to(device)
    return {
        "atom_types":       to_device_list(batch["atom_types"],      device),
        "coords_spherical": to_device_list(batch["coords_spherical"], device),
        "coords_cart":      to_device_list(batch["coords_cart"],      device),
        "edge_index":       to_device_list(batch["edge_index"],       device),
        "edge_attr":        to_device_list(batch["edge_attr"],        device),
        "y":                y,
        "_n_atoms":         n_atoms,
    }


def denormalize(val, n_atoms, y_mean, y_std, normalize, per_atom, device):
    out = val.float()
    if normalize:
        out = out * y_std.to(device) + y_mean.to(device)
    if per_atom:
        out = out * n_atoms.clamp_min(1.0)
    return out


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ─── Profiling epoch ─────────────────────────────────────────────────────────

def profile_train_epoch(model, loader, optimizer, loss_fn, scaler, ema,
                        y_mean, y_std, normalize, per_atom, device,
                        use_amp, phase_reg_weight, grad_clip):
    """
    Ejecuta una época de entrenamiento con instrumentación detallada de timing.

    Devuelve dict con:
      - listas por batch: data_load_ms, forward_ms, backward_ms, batch_sizes
      - resumen: epoch_time_s, avg_forward_ms, avg_backward_ms, avg_data_load_ms
    """
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_load_times = []
    forward_times   = []
    backward_times  = []
    batch_sizes     = []
    total_loss      = 0.0
    total_samples   = 0

    # ── Tiempo de carga del primer batch ─────────────────────────────────────
    sync()
    iter_start = time.perf_counter()
    loader_iter = iter(loader)

    epoch_t0 = time.perf_counter()

    while True:
        # Medir carga de datos (tiempo que tarda el DataLoader en entregar el batch)
        sync()
        t_load_start = time.perf_counter()
        try:
            raw_batch = next(loader_iter)
        except StopIteration:
            break
        sync()
        t_load_end = time.perf_counter()
        data_load_ms = (t_load_end - t_load_start) * 1000.0

        # Preparar batch (transfer a GPU)
        batch = prepare_batch(raw_batch, device, y_mean, y_std, normalize, per_atom)
        bs = batch["y"].shape[0]
        batch_sizes.append(bs)
        data_load_times.append(data_load_ms)

        # ── Forward ───────────────────────────────────────────────────────────
        sync()
        t_fwd_start = time.perf_counter()

        with torch.autocast(device_type=device, enabled=use_amp):
            pred = model(batch)
            pred = torch.as_tensor(pred).float()
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)
            loss = loss_fn(pred, batch["y"])
            phase_reg = getattr(model, "_phase_reg", None)
            if phase_reg is not None and phase_reg_weight > 0:
                loss = loss + phase_reg_weight * phase_reg.float()

        sync()
        t_fwd_end = time.perf_counter()
        forward_times.append((t_fwd_end - t_fwd_start) * 1000.0)

        # ── Backward ──────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        sync()
        t_bwd_start = time.perf_counter()

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        sync()
        t_bwd_end = time.perf_counter()
        backward_times.append((t_bwd_end - t_bwd_start) * 1000.0)

        if ema is not None:
            ema.update(model)

        total_loss    += loss.item() * bs
        total_samples += bs

    sync()
    epoch_time_s = time.perf_counter() - epoch_t0

    return {
        "epoch_time_s":      epoch_time_s,
        "total_samples":     total_samples,
        "throughput_s_per_s": total_samples / max(epoch_time_s, 1e-9),
        "avg_loss":          total_loss / max(total_samples, 1),
        # Timing por batch
        "data_load_ms_per_batch":  data_load_times,
        "forward_ms_per_batch":    forward_times,
        "backward_ms_per_batch":   backward_times,
        "batch_sizes":             batch_sizes,
        # Estadísticas
        "avg_data_load_ms":  float(np.mean(data_load_times))  if data_load_times else 0.0,
        "p95_data_load_ms":  float(np.percentile(data_load_times, 95)) if data_load_times else 0.0,
        "avg_forward_ms":    float(np.mean(forward_times))    if forward_times   else 0.0,
        "p95_forward_ms":    float(np.percentile(forward_times, 95))   if forward_times   else 0.0,
        "avg_backward_ms":   float(np.mean(backward_times))  if backward_times  else 0.0,
        "p95_backward_ms":   float(np.percentile(backward_times, 95))  if backward_times  else 0.0,
        "peak_gpu_mem_mb":   gpu_mem_mb(),
    }


def profile_val_epoch(model, loader, loss_fn, ema,
                      y_mean, y_std, normalize, per_atom, device, use_amp):
    if ema is not None:
        ema.apply(model)

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_loss   = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    sync()
    t0 = time.perf_counter()

    with torch.no_grad():
        for raw_batch in loader:
            batch = prepare_batch(raw_batch, device, y_mean, y_std, normalize, per_atom)
            bs = batch["y"].shape[0]

            with torch.autocast(device_type=device, enabled=use_amp):
                pred = model(batch)
            pred = torch.as_tensor(pred).float()
            if pred.dim() == 1:
                pred = pred.unsqueeze(-1)

            loss = loss_fn(pred, batch["y"])
            total_loss    += loss.item() * bs
            total_samples += bs

            pred_denorm = denormalize(pred, batch["_n_atoms"], y_mean, y_std, normalize, per_atom, device)
            targ_denorm = denormalize(batch["y"], batch["_n_atoms"], y_mean, y_std, normalize, per_atom, device)
            all_preds.append(pred_denorm.cpu())
            all_targets.append(targ_denorm.cpu())

    sync()
    elapsed = time.perf_counter() - t0

    if ema is not None:
        ema.restore(model)

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = evaluate_regression(preds, targets)

    return {
        "val_time_s":        elapsed,
        "val_samples":       total_samples,
        "val_throughput":    total_samples / max(elapsed, 1e-9),
        "avg_loss":          total_loss / max(total_samples, 1),
        "mae_kcal":          metrics.get("mae",  float("nan")),
        "rmse_kcal":         metrics.get("rmse", float("nan")),
        "r2":                metrics.get("r2",   float("nan")),
        "peak_gpu_mem_mb":   gpu_mem_mb(),
    }


# ─── Target statistics ────────────────────────────────────────────────────────

def compute_target_stats(train_ds, per_atom):
    base, idx = _unwrap_subset(train_ds)
    index_array = _as_index_array(idx, len(base))

    target_cols = None
    if hasattr(base, "target_col"):
        target_cols = [base.target_col]
    elif hasattr(base, "target_cols"):
        target_cols = list(base.target_cols)

    if hasattr(base, "df") and target_cols is not None:
        vals = base.df.iloc[index_array][target_cols].values.astype("float32")
        if per_atom and hasattr(base, "num_atoms"):
            n_atoms = np.asarray(base.num_atoms, dtype="float32")[index_array].reshape(-1, 1)
            vals = vals / np.clip(n_atoms, 1.0, None)
        mean = torch.from_numpy(np.mean(vals, axis=0)).float()
        std  = torch.from_numpy(np.std(vals,  axis=0)).float().clamp_min(1e-9)
        return mean, std

    raise RuntimeError("No se pudo calcular estadísticas del target desde el dataset.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline profiler — ComplexPolarTransformer v12")
    parser.add_argument("--config", default="experiments/beta_train_benchmark.yaml")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Número de épocas a perfilar (default: 5)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg.get("amp", {}).get("enabled", True)) and (device == "cuda")

    dataset_cfg = cfg["dataset"]
    model_cfg   = cfg["model"]
    cutoff      = float(model_cfg.get("cutoff", 5.0))
    per_atom    = bool(cfg.get("per_atom_norm", True))
    normalize   = bool(cfg.get("normalize_target", True))

    print(f"\n{'='*70}")
    print(f"  ComplexPolarTransformer v12 — BASELINE PROFILER")
    print(f"  Config : {args.config}")
    print(f"  Device : {device}  AMP: {use_amp}")
    print(f"  Épocas : {args.epochs}")
    print(f"{'='*70}\n")

    # ── Imprimir info CUDA ────────────────────────────────────────────────────
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[GPU] {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GB | "
              f"SM: {props.multi_processor_count} | Cap: {props.major}.{props.minor}")
        print(f"[CUDA] version {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}\n")

    # ── Cargar dataset ────────────────────────────────────────────────────────
    print("[DATASET] Cargando QM9...")
    t_ds = time.perf_counter()
    dataset = QM9SDFDataset(
        sdf_path=dataset_cfg["sdf"],
        csv_path=dataset_cfg["csv"],
        target_col=dataset_cfg.get("target", "u0_atom"),
        max_radius=cutoff,
    )
    t_ds_end = time.perf_counter()
    print(f"[DATASET] Cargado en {t_ds_end - t_ds:.1f}s | {len(dataset)} moléculas\n")

    # ── Split ─────────────────────────────────────────────────────────────────
    n_total = len(dataset)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    n_train, n_val = 110_000, 10_000
    if n_total < 120_000:
        n_train = int(0.8 * n_total)
        n_val   = int(0.1 * n_total)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    print(f"[SPLIT] Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Estadísticas del target ───────────────────────────────────────────────
    y_mean, y_std = compute_target_stats(train_ds, per_atom)
    print(f"[TARGET] mean={y_mean.item():.4f} | std={y_std.item():.4f}\n")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    batch_size  = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 8))
    pin_memory  = (device == "cuda")

    train_dl = build_dataloader(train_ds, batch_size, True,  num_workers, pin_memory)
    val_dl   = build_dataloader(val_ds,   batch_size, False, max(0, num_workers // 2), pin_memory)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = ComplexPolarTransformerBeta(
        in_dim            = int(model_cfg["in_dim"]),
        hidden_dim        = int(model_cfg["hidden_dim"]),
        out_dim           = int(model_cfg.get("out_dim", 1)),
        num_hidden_layers = int(model_cfg.get("num_hidden_layers", 4)),
        num_heads         = int(model_cfg.get("num_heads", 4)),
        num_rbf           = int(model_cfg.get("num_rbf", 64)),
        cutoff            = cutoff,
        edge_dim          = int(model_cfg.get("edge_dim", 4)),
        dropout           = float(model_cfg.get("dropout", 0.0)),
        use_residuals     = bool(model_cfg.get("use_residuals", False)),
        use_layernorm     = bool(model_cfg.get("use_layernorm", False)),
        activation        = str(model_cfg.get("activation", "modrelu")),
        modrelu_init_bias = float(model_cfg.get("modrelu_init_bias", 5.0)),
        modrelu_eps       = float(model_cfg.get("modrelu_eps", 1e-8)),
        use_angular       = bool(model_cfg.get("use_angular", False)),
        num_angle_basis   = int(model_cfg.get("num_angle_basis", 16)),
        angular_scale_init= float(model_cfg.get("angular_scale_init", 0.1)),
    ).to(device)

    param_count   = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"[MODEL] Parámetros: {param_count:,} | Tamaño: {model_size_mb:.2f} MB")

    # ── Optimizer, scaler, EMA ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 3e-4)),
    )
    loss_fn = torch.nn.L1Loss()
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    ema_cfg = cfg.get("ema", {}) or {}
    ema     = EMAHelper(model, decay=float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("enabled", True) else None

    phase_reg_weight = float(cfg.get("phase_reg_weight", 0.0))
    grad_clip        = float(cfg.get("grad_clip", 1.0))

    # ─── Warm-up de CUDA (1 batch, sin contar para baseline) ────────────────
    print("\n[WARMUP] Ejecutando 1 batch de calentamiento CUDA...")
    model.train()
    with torch.no_grad():
        wb = next(iter(train_dl))
        wb = prepare_batch(wb, device, y_mean, y_std, normalize, per_atom)
        with torch.autocast(device_type=device, enabled=use_amp):
            _ = model(wb)
    sync()
    print("[WARMUP] Completado.\n")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ─── Bucle de profiling ──────────────────────────────────────────────────
    epoch_records = []
    print(f"{'─'*70}")
    print(f"{'Ép':>4} | {'Train MAE':>10} | {'Val MAE':>9} | {'Val RMSE':>9} | "
          f"{'R²':>8} | {'Ép(s)':>7} | {'Th(s/s)':>8} | "
          f"{'Fwd(ms)':>8} | {'Bwd(ms)':>8} | {'Load(ms)':>9} | {'GPU(MB)':>8}")
    print(f"{'─'*70}")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_stats = profile_train_epoch(
            model, train_dl, optimizer, loss_fn, scaler, ema,
            y_mean, y_std, normalize, per_atom, device,
            use_amp, phase_reg_weight, grad_clip,
        )

        # Val
        val_stats = profile_val_epoch(
            model, val_dl, loss_fn, ema,
            y_mean, y_std, normalize, per_atom, device, use_amp,
        )

        rec = {
            "epoch":             epoch,
            # Calidad
            "train_l1_norm":     train_stats["avg_loss"],
            "val_mae_kcal":      val_stats["mae_kcal"],
            "val_rmse_kcal":     val_stats["rmse_kcal"],
            "val_r2":            val_stats["r2"],
            # Tiempo y throughput
            "epoch_time_s":      train_stats["epoch_time_s"] + val_stats["val_time_s"],
            "train_time_s":      train_stats["epoch_time_s"],
            "val_time_s":        val_stats["val_time_s"],
            "train_throughput":  train_stats["throughput_s_per_s"],
            "val_throughput":    val_stats["val_throughput"],
            # Timing desagregado
            "avg_data_load_ms":  train_stats["avg_data_load_ms"],
            "p95_data_load_ms":  train_stats["p95_data_load_ms"],
            "avg_forward_ms":    train_stats["avg_forward_ms"],
            "p95_forward_ms":    train_stats["p95_forward_ms"],
            "avg_backward_ms":   train_stats["avg_backward_ms"],
            "p95_backward_ms":   train_stats["p95_backward_ms"],
            # Memoria
            "peak_gpu_mem_train_mb": train_stats["peak_gpu_mem_mb"],
            "peak_gpu_mem_val_mb":   val_stats["peak_gpu_mem_mb"],
        }
        epoch_records.append(rec)

        print(
            f"{epoch:>4} | "
            f"{rec['train_l1_norm']:>10.4f} | "
            f"{rec['val_mae_kcal']:>9.4f} | "
            f"{rec['val_rmse_kcal']:>9.4f} | "
            f"{rec['val_r2']:>8.4f} | "
            f"{rec['epoch_time_s']:>7.1f} | "
            f"{rec['train_throughput']:>8.0f} | "
            f"{rec['avg_forward_ms']:>8.2f} | "
            f"{rec['avg_backward_ms']:>8.2f} | "
            f"{rec['avg_data_load_ms']:>9.2f} | "
            f"{rec['peak_gpu_mem_train_mb']:>8.1f}"
        )

    print(f"{'─'*70}\n")

    # ─── Estadísticas de estabilización (promedio épocas 2+) ─────────────────
    stable = epoch_records[1:] if len(epoch_records) > 1 else epoch_records

    def avg(key):
        vals = [r[key] for r in stable if isinstance(r[key], float) and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "model": {
            "params":    param_count,
            "size_mb":   model_size_mb,
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "num_layers": int(model_cfg.get("num_hidden_layers", 4)),
            "num_heads":  int(model_cfg.get("num_heads", 4)),
            "num_rbf":    int(model_cfg.get("num_rbf", 64)),
            "use_angular": bool(model_cfg.get("use_angular", False)),
            "num_angle_basis": int(model_cfg.get("num_angle_basis", 16)),
            "cutoff_A":   cutoff,
            "activation": model_cfg.get("activation", "modrelu"),
        },
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "cuda_version": torch.version.cuda if device == "cuda" else None,
            "use_amp": use_amp,
        },
        "dataset": {
            "n_train": n_total if n_total < 120_000 else 110_000,
            "n_val":   n_total if n_total < 120_000 else 10_000,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "target": dataset_cfg.get("target", "u0_atom"),
            "y_mean": float(y_mean.item()),
            "y_std":  float(y_std.item()),
        },
        "training": {
            "use_ema":    ema is not None,
            "use_amp":    use_amp,
            "grad_clip":  grad_clip,
            "phase_reg_weight": phase_reg_weight,
        },
        "baseline_metrics": {
            # Promedio de épocas estables (época 2 en adelante)
            "avg_epoch_time_s":      avg("epoch_time_s"),
            "avg_train_time_s":      avg("train_time_s"),
            "avg_val_time_s":        avg("val_time_s"),
            "avg_train_throughput":  avg("train_throughput"),
            "avg_val_throughput":    avg("val_throughput"),
            "avg_data_load_ms":      avg("avg_data_load_ms"),
            "p95_data_load_ms":      avg("p95_data_load_ms"),
            "avg_forward_ms":        avg("avg_forward_ms"),
            "p95_forward_ms":        avg("p95_forward_ms"),
            "avg_backward_ms":       avg("avg_backward_ms"),
            "p95_backward_ms":       avg("p95_backward_ms"),
            "avg_peak_gpu_mem_mb":   avg("peak_gpu_mem_train_mb"),
            # MAE de última época del profiler
            "val_mae_epoch_final":   epoch_records[-1]["val_mae_kcal"],
            "val_rmse_epoch_final":  epoch_records[-1]["val_rmse_kcal"],
            "val_r2_epoch_final":    epoch_records[-1]["val_r2"],
        },
        "epoch_records": epoch_records,
    }

    # ─── Guardar JSON ─────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    json_path = os.path.join("logs", "baseline_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OUTPUT] Métricas guardadas en: {json_path}")

    # ─── Generar BASELINE.md ──────────────────────────────────────────────────
    bm = summary["baseline_metrics"]
    hw = summary["hardware"]
    md = summary["model"]
    ds = summary["dataset"]

    lines = [
        "# ComplexPolarTransformer v12 — Línea Base",
        "",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Config: `{args.config}`  ",
        f"Épocas perfiladas: {args.epochs}  ",
        f"Épocas promediadas (estables): {len(stable)}",
        "",
        "---",
        "",
        "## Hardware",
        "",
        f"| Parámetro | Valor |",
        f"|-----------|-------|",
        f"| GPU | {hw['gpu_name']} |",
        f"| CUDA | {hw['cuda_version']} |",
        f"| AMP (mixed precision) | {hw['use_amp']} |",
        "",
        "## Modelo",
        "",
        f"| Parámetro | Valor |",
        f"|-----------|-------|",
        f"| Parámetros totales | {md['params']:,} |",
        f"| Tamaño en disco | {md['size_mb']:.2f} MB |",
        f"| hidden_dim | {md['hidden_dim']} |",
        f"| num_hidden_layers | {md['num_layers']} |",
        f"| num_heads | {md['num_heads']} |",
        f"| num_rbf | {md['num_rbf']} |",
        f"| use_angular | {md['use_angular']} |",
        f"| num_angle_basis | {md['num_angle_basis']} |",
        f"| cutoff | {md['cutoff_A']} Å |",
        f"| activación | {md['activation']} |",
        "",
        "## Dataset",
        "",
        f"| Parámetro | Valor |",
        f"|-----------|-------|",
        f"| Moléculas train | {ds['n_train']:,} |",
        f"| Moléculas val | {ds['n_val']:,} |",
        f"| Batch size | {ds['batch_size']} |",
        f"| DataLoader workers | {ds['num_workers']} |",
        f"| Target | {ds['target']} |",
        f"| y_mean | {ds['y_mean']:.6f} kcal/mol/átomo |",
        f"| y_std | {ds['y_std']:.6f} kcal/mol/átomo |",
        "",
        "## Métricas de Rendimiento (promedio épocas estables)",
        "",
        "### Throughput y Tiempo de Época",
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| **Tiempo por época (train+val)** | **{bm['avg_epoch_time_s']:.1f} s ({bm['avg_epoch_time_s']/60:.1f} min)** |",
        f"| Tiempo de entrenamiento | {bm['avg_train_time_s']:.1f} s |",
        f"| Tiempo de validación | {bm['avg_val_time_s']:.1f} s |",
        f"| **Throughput train** | **{bm['avg_train_throughput']:.0f} muestras/s** |",
        f"| Throughput val | {bm['avg_val_throughput']:.0f} muestras/s |",
        "",
        "### Desglose de Tiempo por Batch",
        "",
        f"| Operación | Promedio | P95 |",
        f"|-----------|----------|-----|",
        f"| Carga de datos (DataLoader) | {bm['avg_data_load_ms']:.2f} ms | {bm['p95_data_load_ms']:.2f} ms |",
        f"| Forward pass | {bm['avg_forward_ms']:.2f} ms | {bm['p95_forward_ms']:.2f} ms |",
        f"| Backward pass | {bm['avg_backward_ms']:.2f} ms | {bm['p95_backward_ms']:.2f} ms |",
        "",
        "### Uso de Memoria GPU",
        "",
        f"| Métrica | Valor |",
        f"|---------|-------|",
        f"| **Pico GPU (train)** | **{bm['avg_peak_gpu_mem_mb']:.1f} MB** |",
        "",
        "## Calidad del Modelo",
        "",
        f"> ⚠️  MAE medido después de solo {args.epochs} épocas — NO es el MAE de convergencia.",
        f"> Para la línea base final de MAE ejecutar `main_train_benchmark.py` completo.",
        "",
        f"| Métrica | Época {args.epochs} |",
        f"|---------|-------------|",
        f"| **Val MAE** | **{bm['val_mae_epoch_final']:.4f} kcal/mol** |",
        f"| Val RMSE | {bm['val_rmse_epoch_final']:.4f} kcal/mol |",
        f"| Val R² | {bm['val_r2_epoch_final']:.6f} |",
        "",
        "### Curva MAE por Época (profiler)",
        "",
        "| Época | Val MAE (kcal/mol) | Val RMSE | Val R² | Ép (s) | Train (s/s) | Fwd (ms) | Bwd (ms) | Load (ms) | GPU (MB) |",
        "|------:|------------------:|----------|--------|-------:|------------:|----------|----------|-----------|----------|",
    ]

    for r in epoch_records:
        lines.append(
            f"| {r['epoch']} "
            f"| {r['val_mae_kcal']:.4f} "
            f"| {r['val_rmse_kcal']:.4f} "
            f"| {r['val_r2']:.4f} "
            f"| {r['epoch_time_s']:.1f} "
            f"| {r['train_throughput']:.0f} "
            f"| {r['avg_forward_ms']:.2f} "
            f"| {r['avg_backward_ms']:.2f} "
            f"| {r['avg_data_load_ms']:.2f} "
            f"| {r['peak_gpu_mem_train_mb']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Referencias Benchmark QM9 (U₀, kcal/mol)",
        "",
        "| Modelo | MAE |",
        "|--------|-----|",
        "| SchNet | 0.313 |",
        "| DimeNet++ | 0.215 |",
        "| PaiNN | 0.224 |",
        "| NequIP | 0.038 |",
        "| **Este modelo (baseline)** | *ver entrenamiento completo* |",
        "",
        "---",
        "",
        f"*Generado por `run_baseline_profiler.py`*",
    ]

    md_path = os.path.join("logs", "BASELINE.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OUTPUT] Resumen legible en: {md_path}")
    print(f"\n{'='*70}")
    print(f"  BASELINE COMPLETADO")
    print(f"  Época promedio : {bm['avg_epoch_time_s']:.1f}s = {bm['avg_epoch_time_s']/60:.1f} min")
    print(f"  Throughput     : {bm['avg_train_throughput']:.0f} s/s (train) | {bm['avg_val_throughput']:.0f} s/s (val)")
    print(f"  Forward/batch  : {bm['avg_forward_ms']:.2f} ms (avg) | {bm['p95_forward_ms']:.2f} ms (p95)")
    print(f"  Backward/batch : {bm['avg_backward_ms']:.2f} ms (avg) | {bm['p95_backward_ms']:.2f} ms (p95)")
    print(f"  Load/batch     : {bm['avg_data_load_ms']:.2f} ms (avg) | {bm['p95_data_load_ms']:.2f} ms (p95)")
    print(f"  GPU pico       : {bm['avg_peak_gpu_mem_mb']:.1f} MB")
    print(f"  Val MAE@ep{args.epochs}   : {bm['val_mae_epoch_final']:.4f} kcal/mol")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
