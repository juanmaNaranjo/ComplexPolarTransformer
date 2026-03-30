import argparse
import os
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from core.dataset import QM9SDFDataset
from models.complex_model_beta import ComplexPolarTransformerBeta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import pandas as pd


def collate_mol(batch):
    return {
        "coords_cart": [item["coords_cart"] for item in batch],
        "coords_spherical": [item["coords_spherical"] for item in batch],
        "atom_types": [item["atom_types"] for item in batch],
        "edge_index": [item["edge_index"] for item in batch],
        "edge_attr": [item["edge_attr"] for item in batch],
        "y": torch.stack([item["y"] for item in batch])
    }


#def load_checkpoint(model_path, device):
 #   if not os.path.exists(model_path):
  #      raise FileNotFoundError(f"Checkpoint not found: {model_path}")
   # ckpt = torch.load(model_path, map_location=device)
    # If saved as full checkpoint dict
    #if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
     #   state_dict = ckpt["model_state_dict"]
    #else:
     #   state_dict = ckpt
    #return ckpt, state_dict

def load_checkpoint(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]   # 🔥 ESTA ES LA CLAVE
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    return ckpt, state_dict


def build_model_from_ckpt(ckpt, state_dict, defaults=None):
    # defaults: dict with fallback in_dim, hidden_dim, out_dim
    defaults = defaults or {"in_dim": 5, "hidden_dim": 256, "out_dim": 1}
    model_cfg = defaults.copy()
    if isinstance(ckpt, dict) and "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
        h = ckpt["hparams"]
        if "model" in h and isinstance(h["model"], dict):
            model_cfg.update(h["model"])
    # Ensure keys exist
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
    # Try strict load first, then fallback to non-strict and report missing/unexpected keys
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("Warning: strict state_dict load failed:", e)
        res = model.load_state_dict(state_dict, strict=False)
        # res is an _IncompatibleKeys object with missing_keys and unexpected_keys
        missing = getattr(res, 'missing_keys', None)
        unexpected = getattr(res, 'unexpected_keys', None)
        if missing:
            print("Missing keys when loading state_dict:", missing)
        if unexpected:
            print("Unexpected keys in state_dict:", unexpected)
        print("Proceeding with partially loaded model (uninitialized params remain).")
    return model


def predict(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    if not os.path.exists(args.sdf):
        raise FileNotFoundError(f"SDF not found: {args.sdf}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    dataset = QM9SDFDataset(sdf_path=args.sdf, csv_path=args.csv, target_col=args.target)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mol,
                            num_workers=args.num_workers, pin_memory=(device=="cuda"))

    # Load checkpoint and build model
    ckpt, state_dict = load_checkpoint(args.model, device)
    model = build_model_from_ckpt(ckpt, state_dict)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    # y normalization stats if available
    y_mean = float(ckpt.get("y_mean", 0.0)) if isinstance(ckpt, dict) else 0.0
    y_std = float(ckpt.get("y_std", 1.0)) if isinstance(ckpt, dict) else 1.0

    y_trues = []
    y_preds = []

    indices = []
    idx = 0
    for batch in tqdm(dataloader, desc="Predict"):
        # move tensors to device and update batch
        feats = [f.to(device) for f in batch["atom_types"]]
        coords = [c.to(device) for c in batch.get("coords_spherical", [])]
        batch["atom_types"] = feats
        batch["coords_spherical"] = coords

        with torch.no_grad():
            preds = model(batch).squeeze(-1)

        # desnormalizar si corresponde
        preds = preds * y_std + y_mean

        target = batch["y"].cpu().numpy()
        preds_np = preds.cpu().numpy()

        batch_size = len(target)
        y_trues.extend(target.tolist())
        y_preds.extend(preds_np.tolist())
        indices.extend(list(range(idx, idx + batch_size)))
        idx += batch_size

        if args.dry_run and idx >= args.dry_run:
            break

    if device == "cuda":
        torch.cuda.synchronize()

    inference_total_sec = time.perf_counter() - t0

    # Métricas (filtrar NaNs)
    y_true_arr = np.array(y_trues)
    y_pred_arr = np.array(y_preds)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]

    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    mae_val = mean_absolute_error(y_true_arr, y_pred_arr)
    r2 = r2_score(y_true_arr, y_pred_arr)

    num_samples = int(len(y_true_arr))
    inference_ms_per_sample = (inference_total_sec / max(num_samples, 1)) * 1000.0
    samples_per_sec = num_samples / max(inference_total_sec, 1e-9)

    peak_gpu_mem_mb = None
    if device == "cuda":
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae_val:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Inference total time (s): {inference_total_sec:.6f}")
    print(f"Latency per sample (ms): {inference_ms_per_sample:.6f}")
    print(f"Throughput (samples/s): {samples_per_sec:.6f}")
    print(f"Model params: {param_count:,}")
    print(f"Model size (MB): {model_size_mb:.4f}")
    if peak_gpu_mem_mb is not None:
        print(f"Peak GPU memory (MB): {peak_gpu_mem_mb:.4f}")

    # Guardar predicciones
    out_df = pd.DataFrame({"idx": indices[:len(y_pred_arr)], "y_true": y_true_arr, "y_pred": y_pred_arr})
    os.makedirs(os.path.dirname(args.output), exist_ok=True) if args.output and os.path.dirname(args.output) else None
    out_df.to_csv(args.output, index=False)

    # Guardar metadatos
    meta = {
        "model": args.model,
        "sdf": args.sdf,
        "csv": args.csv,
        "batch_size": args.batch_size,
        "num_samples": int(len(y_true_arr)),
        "metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae_val),
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
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    meta_path = args.output.replace('.csv', '.meta.json') if args.output.endswith('.csv') else args.output + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_arr, y_pred_arr, alpha=0.5)
    mn, mx = np.min(y_true_arr), np.max(y_true_arr)
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores reales')
    plt.savefig(args.plot)
    if args.show:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(description='Predict using ComplexPolarTransformer')
    p.add_argument('--sdf', required=True, help='Path to qm9.sdf')
    p.add_argument('--csv', required=True, help='Path to CSV with targets')
    p.add_argument('--target', default='u0', help='Target column name in CSV')
    p.add_argument('--model', required=True, help='Path to model checkpoint')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default=None, help='cuda or cpu (auto if omitted)')
    p.add_argument('--output', default='results/predictions.csv', help='Output CSV path')
    p.add_argument('--plot', default='results/pred_vs_real.png', help='Output plot path')
    p.add_argument('--show', action='store_true', help='Show plot interactively')
    p.add_argument('--dry-run', type=int, default=0, help='Stop after N samples (for quick debug)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)
