import argparse
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from core.dataset import QM9SDFDataset
from models.complex_model_beta import ComplexPolarTransformerBeta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pandas as pd


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.509474  # kcal/mol
EV_TO_MEV = 1000.0


def collate_mol(batch):
    return {
        "coords_cart": [item["coords_cart"] for item in batch],
        "coords_spherical": [item["coords_spherical"] for item in batch],
        "atom_types": [item["atom_types"] for item in batch],
        "edge_index": [item["edge_index"] for item in batch],
        "edge_attr": [item["edge_attr"] for item in batch],
        "y": torch.stack([item["y"] for item in batch]),
    }


def load_checkpoint(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    return ckpt, state_dict


def build_model_from_ckpt(ckpt, state_dict, defaults=None):
    defaults = defaults or {"in_dim": 5, "hidden_dim": 256, "out_dim": 1}
    model_cfg = defaults.copy()
    if isinstance(ckpt, dict) and "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
        h = ckpt["hparams"]
        if "model" in h and isinstance(h["model"], dict):
            model_cfg.update(h["model"])

    in_dim = int(model_cfg.get("in_dim", defaults["in_dim"]))
    hidden_dim = int(model_cfg.get("hidden_dim", defaults["hidden_dim"]))
    out_dim = int(model_cfg.get("out_dim", defaults["out_dim"]))
    num_hidden_layers = int(model_cfg.get("num_hidden_layers", 1))
    dropout = float(model_cfg.get("dropout", 0.0))
    use_residuals = bool(model_cfg.get("use_residuals", False))
    use_layernorm = bool(model_cfg.get("use_layernorm", False))

    model = ComplexPolarTransformerBeta(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        use_residuals=use_residuals,
        use_layernorm=use_layernorm,
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("Warning: strict state_dict load failed:", e)
        res = model.load_state_dict(state_dict, strict=False)
        missing = getattr(res, "missing_keys", None)
        unexpected = getattr(res, "unexpected_keys", None)
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)
        print("Proceeding with partially loaded model.")

    return model


def load_split(split_file, split_name):
    if not split_file:
        return None
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        s = json.load(f)
    key = f"{split_name}_idx"
    if key not in s:
        raise KeyError(f"Split file no tiene '{key}'. Keys: {list(s.keys())}")
    return s[key], s


def convert_mae_units(mae_raw, unit):
    """
    Convierte MAE a meV y kcal/mol desde la unidad base.
    unit: 'hartree' | 'ev' | 'kcal' | 'mev'

    NOTA para QM9: u0_atom en el CSV estándar está en kcal/mol.
    Usar --unit kcal para obtener métricas correctas.
    """
    if unit == "hartree":
        mae_ev   = mae_raw * HARTREE_TO_EV
        mae_mev  = mae_ev  * EV_TO_MEV
        mae_kcal = mae_raw * HARTREE_TO_KCAL_MOL
        return mae_mev, mae_kcal
    if unit == "ev":
        mae_mev  = mae_raw * EV_TO_MEV
        mae_kcal = (mae_raw / HARTREE_TO_EV) * HARTREE_TO_KCAL_MOL
        return mae_mev, mae_kcal
    if unit == "mev":
        mae_mev  = mae_raw
        mae_kcal = ((mae_raw / EV_TO_MEV) / HARTREE_TO_EV) * HARTREE_TO_KCAL_MOL
        return mae_mev, mae_kcal
    if unit == "kcal":
        # u0_atom ya está en kcal/mol — conversión directa sin factor Hartree
        mae_kcal = mae_raw
        KCAL_TO_EV = 0.043363
        mae_mev  = mae_kcal * KCAL_TO_EV * EV_TO_MEV
        return mae_mev, mae_kcal
    return None, None


def predict(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.sdf):
        raise FileNotFoundError(f"SDF not found: {args.sdf}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    # Dataset completo
    dataset_full = QM9SDFDataset(sdf_path=args.sdf, csv_path=args.csv, target_col=args.target)

    # Aplicar split si existe
    split_meta = None
    if args.split_file:
        idxs, split_meta = load_split(args.split_file, args.split)
        dataset = Subset(dataset_full, idxs)
        print(f"[INFO] Split '{args.split}' aplicado: {len(dataset)} muestras")
    else:
        dataset = dataset_full
        print(f"[INFO] Sin split_file: usando dataset completo ({len(dataset)} muestras)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_mol,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Load checkpoint and build model
    ckpt, state_dict = load_checkpoint(args.model, device)
    model = build_model_from_ckpt(ckpt, state_dict)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    # y normalization stats if available (tensor-safe)
    if isinstance(ckpt, dict) and "y_mean" in ckpt and "y_std" in ckpt:
        y_mean = torch.as_tensor(ckpt["y_mean"]).float().view(-1).to(device)
        y_std = torch.as_tensor(ckpt["y_std"]).float().view(-1).to(device)
        # single-target expected
        if y_mean.numel() != 1 or y_std.numel() != 1:
            # Si el checkpoint es multi-target, toma el primer target (o falla)
            if args.force_first_target:
                y_mean = y_mean[:1]
                y_std = y_std[:1]
            else:
                raise ValueError("Checkpoint parece multi-target. Usa --force-first-target o evalúa multi-target.")
    else:
        y_mean = torch.zeros(1, dtype=torch.float32, device=device)
        y_std = torch.ones(1, dtype=torch.float32, device=device)

    y_trues = []
    y_preds = []
    indices = []
    seen = 0

    for batch in tqdm(dataloader, desc="Predict"):
        # mover al device
        batch["atom_types"] = [f.to(device) for f in batch["atom_types"]]
        batch["coords_spherical"] = [c.to(device) for c in batch.get("coords_spherical", [])]
        batch["coords_cart"] = [cc.to(device) for cc in batch.get("coords_cart", [])]
        batch["edge_index"] = [ei.to(device) for ei in batch.get("edge_index", [])]
        batch["edge_attr"] = [ea.to(device) for ea in batch.get("edge_attr", [])]

        with torch.no_grad():
            preds = model(batch)
            preds = torch.as_tensor(preds).float()
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)

        # desnormalizar
        preds = preds * y_std + y_mean

        target = batch["y"].cpu().numpy()  # (B,1)
        preds_np = preds.cpu().numpy()     # (B,1)

        bs = target.shape[0]

        if args.dry_run and (seen + bs) > args.dry_run:
            keep = args.dry_run - seen
            if keep <= 0:
                break
            target = target[:keep]
            preds_np = preds_np[:keep]
            bs = keep

        y_trues.append(target.reshape(-1))
        y_preds.append(preds_np.reshape(-1))
        indices.extend(list(range(seen, seen + bs)))
        seen += bs

        if args.dry_run and seen >= args.dry_run:
            break

    if device == "cuda":
        torch.cuda.synchronize()

    inference_total_sec = time.perf_counter() - t0

    y_true_arr = np.concatenate(y_trues, axis=0) if y_trues else np.array([])
    y_pred_arr = np.concatenate(y_preds, axis=0) if y_preds else np.array([])

    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]

    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = float(np.sqrt(mse))
    mae_val = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)

    num_samples = int(len(y_true_arr))
    inference_ms_per_sample = (inference_total_sec / max(num_samples, 1)) * 1000.0
    samples_per_sec = num_samples / max(inference_total_sec, 1e-9)

    peak_gpu_mem_mb = None
    if device == "cuda":
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Conversion a unidades benchmark
    mae_mev, mae_kcal = convert_mae_units(mae_val, args.unit)

    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f} [{args.unit}]")
    print(f"MAE:  {mae_val:.6f} [{args.unit}]")
    if mae_mev is not None and mae_kcal is not None:
        print(f"MAE:  {mae_kcal:.4f} kcal/mol")
        print(f"MAE:  {mae_mev:.2f} meV")
        print(f"--- Benchmark QM9 u0_atom ---")
        print(f"  NequIP:     0.0420 kcal/mol")
        print(f"  TensorNet:  0.0580 kcal/mol")
        print(f"  PaiNN:      0.1490 kcal/mol")
        print(f"  SchNet:     0.3130 kcal/mol")
        print(f"  MPNN:       0.3550 kcal/mol")
        print(f"  Este modelo:{mae_kcal:8.4f} kcal/mol  (factor vs SchNet: {mae_kcal/0.3130:.1f}x)")
    print(f"R2:   {r2:.6f}")
    print(f"Inference total time (s): {inference_total_sec:.6f}")
    print(f"Latency per sample (ms): {inference_ms_per_sample:.6f}")
    print(f"Throughput (samples/s): {samples_per_sec:.6f}")
    print(f"Model params: {param_count:,}")
    print(f"Model size (MB): {model_size_mb:.4f}")
    if peak_gpu_mem_mb is not None:
        print(f"Peak GPU memory (MB): {peak_gpu_mem_mb:.4f}")

    # Guardar predicciones
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
        out_df = pd.DataFrame({"idx": indices[:len(y_pred_arr)], "y_true": y_true_arr, "y_pred": y_pred_arr})
        out_df.to_csv(args.output, index=False)

        # Guardar metadatos
        meta = {
            "model": args.model,
            "sdf": args.sdf,
            "csv": args.csv,
            "target": args.target,
            "split_file": args.split_file,
            "split": args.split,
            "batch_size": args.batch_size,
            "num_samples": int(len(y_true_arr)),
            "metrics": {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae_raw": float(mae_val),
                "mae_mev": float(mae_mev) if mae_mev is not None else None,
                "mae_kcal_mol": float(mae_kcal) if mae_kcal is not None else None,
                "r2": float(r2),
            },
            "performance": {
                "inference_total_sec": float(inference_total_sec),
                "inference_ms_per_sample": float(inference_ms_per_sample),
                "samples_per_sec": float(samples_per_sec),
                "param_count": int(param_count),
                "model_size_mb": float(model_size_mb),
                "peak_gpu_mem_mb": float(peak_gpu_mem_mb) if peak_gpu_mem_mb is not None else None,
            },
            "split_meta": split_meta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        meta_path = args.output.replace(".csv", ".meta.json") if args.output.endswith(".csv") else args.output + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # Plot
    if args.plot:
        os.makedirs(os.path.dirname(args.plot), exist_ok=True) if os.path.dirname(args.plot) else None
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_arr, y_pred_arr, alpha=0.5)
        mn, mx = float(np.min(y_true_arr)), float(np.max(y_true_arr))
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.xlabel("Valores reales")
        plt.ylabel("Predicciones")
        plt.title("Predicciones vs Valores reales")
        plt.savefig(args.plot, dpi=200, bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Predict using ComplexPolarTransformer (benchmark-ready)")
    p.add_argument("--sdf", required=True, help="Path to qm9.sdf")
    p.add_argument("--csv", required=True, help="Path to CSV with targets (aligned to SDF)")
    p.add_argument("--target", default="u0", help="Target column name in CSV (benchmark: u0)")
    p.add_argument("--model", required=True, help="Path to model checkpoint")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    p.add_argument("--output", default="results/predictions.csv", help="Output CSV path")
    p.add_argument("--plot", default="results/pred_vs_real.png", help="Output plot path")
    p.add_argument("--show", action="store_true", help="Show plot interactively")
    p.add_argument("--dry-run", type=int, default=0, help="Stop after N samples (for quick debug)")

    # Benchmark extras
    p.add_argument("--split-file", default=None, help="Path to split_seed*.json from training (optional)")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to evaluate if split-file is given")
    p.add_argument("--unit", default="hartree", choices=["hartree", "ev", "mev", "kcal"], help="Unidad de u0 en tu CSV (QM9 típico: hartree)")
    p.add_argument("--force-first-target", action="store_true", help="Si el checkpoint es multi-target, usar solo el primer target para evaluar")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args)