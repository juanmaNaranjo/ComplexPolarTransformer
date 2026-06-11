import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.dataset import QM9SDFDataset
from models.complex_model_beta import ComplexPolarTransformerBeta


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL_MOL = 627.509474
EV_TO_MEV = 1000.0


def collate_mol(batch):
    return {
        "coords_cart": [item["coords_cart"] for item in batch],
        "coords_spherical": [item["coords_spherical"] for item in batch],
        "atom_types": [item["atom_types"] for item in batch],
        "edge_index": [item["edge_index"] for item in batch],
        "edge_attr": [item["edge_attr"] for item in batch],
        "y": torch.stack([item["y"] for item in batch]),
        "num_atoms": torch.tensor([item["num_atoms"] for item in batch], dtype=torch.float32),
        "num_edges": torch.tensor([item["num_edges"] for item in batch], dtype=torch.long),
        "original_idx": torch.tensor([item.get("original_idx", -1) for item in batch], dtype=torch.long),
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


def extract_model_config(ckpt, defaults=None):
    defaults = defaults or {
        "in_dim": 5,
        "hidden_dim": 256,
        "out_dim": 1,
        "num_hidden_layers": 3,
        "num_rbf": 50,
        "cutoff": 5.0,
        "edge_dim": 4,
        "dropout": 0.0,
        "use_residuals": False,
        "use_layernorm": False,
        "activation": "modrelu",
        "modrelu_init_bias": -0.1,
        "modrelu_eps": 1e-8,
        "use_angular": False,
        "num_angle_basis": 16,
        "angular_scale_init": 0.1,
    }
    model_cfg = defaults.copy()

    if isinstance(ckpt, dict) and "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
        hparams = ckpt["hparams"]
        if "model" in hparams and isinstance(hparams["model"], dict):
            model_cfg.update(hparams["model"])

    return model_cfg


def build_model_from_ckpt(ckpt, state_dict, allow_partial_load=False):
    model_cfg = extract_model_config(ckpt)

    model = ComplexPolarTransformerBeta(
        in_dim=int(model_cfg.get("in_dim", 5)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        out_dim=int(model_cfg.get("out_dim", 1)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 3)),
        num_rbf=int(model_cfg.get("num_rbf", 50)),
        cutoff=float(model_cfg.get("cutoff", 5.0)),
        edge_dim=int(model_cfg.get("edge_dim", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_residuals=bool(model_cfg.get("use_residuals", False)),
        use_layernorm=bool(model_cfg.get("use_layernorm", False)),
        activation=str(model_cfg.get("activation", "modrelu")),
        modrelu_init_bias=float(model_cfg.get("modrelu_init_bias", -0.1)),
        modrelu_eps=float(model_cfg.get("modrelu_eps", 1e-8)),
        use_angular=bool(model_cfg.get("use_angular", False)),
        num_angle_basis=int(model_cfg.get("num_angle_basis", 16)),
        angular_scale_init=float(model_cfg.get("angular_scale_init", 0.1)),
    )

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if not allow_partial_load:
            raise RuntimeError(
                "No fue posible cargar el checkpoint con strict=True. "
                "Esto normalmente indica que la arquitectura del YAML/checkpoint no coincide. "
                "Usa --allow-partial-load solo para depuración, no para reportar métricas oficiales."
            ) from exc
        res = model.load_state_dict(state_dict, strict=False)
        print("[WARNING] Checkpoint cargado parcialmente. No usar estas métricas como resultado oficial.")
        if getattr(res, "missing_keys", None):
            print("Missing keys:", res.missing_keys)
        if getattr(res, "unexpected_keys", None):
            print("Unexpected keys:", res.unexpected_keys)

    return model, model_cfg


def load_split(split_file, split_name):
    if not split_file:
        return None, None
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    if split_name == "all":
        return None, split_meta

    key = f"{split_name}_idx"
    if key not in split_meta:
        raise KeyError(f"Split file no tiene '{key}'. Keys: {list(split_meta.keys())}")
    return split_meta[key], split_meta


def convert_mae_units(mae_raw, unit):
    """Convierte MAE a meV y kcal/mol desde la unidad base indicada."""
    if unit == "hartree":
        mae_ev = mae_raw * HARTREE_TO_EV
        return mae_ev * EV_TO_MEV, mae_raw * HARTREE_TO_KCAL_MOL
    if unit == "ev":
        return mae_raw * EV_TO_MEV, (mae_raw / HARTREE_TO_EV) * HARTREE_TO_KCAL_MOL
    if unit == "mev":
        return mae_raw, ((mae_raw / EV_TO_MEV) / HARTREE_TO_EV) * HARTREE_TO_KCAL_MOL
    if unit == "kcal":
        mae_kcal = mae_raw
        mae_mev = ((mae_raw / HARTREE_TO_KCAL_MOL) * HARTREE_TO_EV) * EV_TO_MEV
        return mae_mev, mae_kcal
    return None, None


def tensor_stat_from_ckpt(ckpt, key, default, device):
    if isinstance(ckpt, dict) and key in ckpt:
        return torch.as_tensor(ckpt[key]).float().view(-1).to(device)
    return torch.as_tensor(default).float().view(-1).to(device)


def predict(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.sdf):
        raise FileNotFoundError(f"SDF not found: {args.sdf}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    ckpt, state_dict = load_checkpoint(args.model, device)
    model, model_cfg = build_model_from_ckpt(
        ckpt,
        state_dict,
        allow_partial_load=args.allow_partial_load,
    )
    model.to(device)
    model.eval()

    cutoff = float(model_cfg.get("cutoff", args.cutoff if args.cutoff is not None else 5.0))
    num_rbf = int(model_cfg.get("num_rbf", 50))
    print(
        f"[INFO] Modelo reconstruido con cutoff={cutoff:.3f} Å | "
        f"num_rbf={num_rbf} | activation={model_cfg.get('activation', 'modrelu')} | "
        f"use_angular={model_cfg.get('use_angular', False)} | "
        f"num_angle_basis={model_cfg.get('num_angle_basis', 16)}"
    )

    dataset_full = QM9SDFDataset(
        sdf_path=args.sdf,
        csv_path=args.csv,
        target_col=args.target,
        max_radius=cutoff,
    )

    split_idxs, split_meta = load_split(args.split_file, args.split)

    if split_meta and split_meta.get("target") and split_meta.get("target") != args.target:
        print(f"[WARNING] split_file fue creado para target={split_meta.get('target')}, pero args.target={args.target}")

    # Si el entrenamiento usó sample_size, primero reconstruir ese subconjunto y luego aplicar split.
    sample_indices = split_meta.get("sample_indices") if isinstance(split_meta, dict) else None
    dataset_for_split = dataset_full
    if sample_indices:
        dataset_for_split = Subset(dataset_full, sample_indices)
        print(f"[INFO] Aplicado sample_indices del split_file: {len(sample_indices)} muestras")

    if split_idxs is not None:
        dataset = Subset(dataset_for_split, split_idxs)
        print(f"[INFO] Split '{args.split}' aplicado: {len(dataset)} muestras")
    else:
        dataset = dataset_for_split
        print(f"[INFO] Evaluando dataset completo: {len(dataset)} muestras")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_mol,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    y_mean = tensor_stat_from_ckpt(ckpt, "y_mean", [0.0], device)
    y_std = tensor_stat_from_ckpt(ckpt, "y_std", [1.0], device)
    if y_mean.numel() != 1 or y_std.numel() != 1:
        if args.force_first_target:
            y_mean = y_mean[:1]
            y_std = y_std[:1]
        else:
            raise ValueError("Checkpoint parece multi-target. Usa --force-first-target solo para depuración.")

    normalize_target = bool(ckpt.get("normalize_target", True)) if isinstance(ckpt, dict) else True
    per_atom_norm = bool(ckpt.get("per_atom_norm", False)) if isinstance(ckpt, dict) else False

    print(f"[INFO] normalize_target={normalize_target} | per_atom_norm={per_atom_norm}")
    if per_atom_norm:
        print("[INFO] Se desnormaliza per-atom multiplicando por N_atoms por molécula.")

    y_trues, y_preds, out_indices, out_original_indices = [], [], [], []
    seen = 0
    t0 = time.perf_counter()

    for batch in tqdm(dataloader, desc="Predict"):
        batch["atom_types"] = [f.to(device) for f in batch["atom_types"]]
        batch["coords_spherical"] = [c.to(device) for c in batch["coords_spherical"]]
        batch["coords_cart"] = [cc.to(device) for cc in batch["coords_cart"]]
        batch["edge_index"] = [ei.to(device) for ei in batch["edge_index"]]
        batch["edge_attr"] = [ea.to(device) for ea in batch["edge_attr"]]

        with torch.no_grad():
            preds = model(batch)
            preds = torch.as_tensor(preds).float()
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)

        if normalize_target:
            preds = preds * y_std + y_mean

        if per_atom_norm:
            n_atoms = torch.as_tensor(batch["num_atoms"], dtype=torch.float32, device=device).view(-1, 1)
            preds = preds * n_atoms.clamp_min(1.0)

        target = batch["y"].cpu().numpy().reshape(-1)
        preds_np = preds.cpu().numpy().reshape(-1)
        original_idx = batch["original_idx"].cpu().numpy().reshape(-1)

        bs = target.shape[0]
        if args.dry_run and (seen + bs) > args.dry_run:
            keep = args.dry_run - seen
            if keep <= 0:
                break
            target = target[:keep]
            preds_np = preds_np[:keep]
            original_idx = original_idx[:keep]
            bs = keep

        y_trues.append(target)
        y_preds.append(preds_np)
        out_indices.extend(range(seen, seen + bs))
        out_original_indices.extend(original_idx.tolist())
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
    out_indices = list(np.asarray(out_indices)[mask])
    out_original_indices = list(np.asarray(out_original_indices)[mask])

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

    mae_mev, mae_kcal = convert_mae_units(mae_val, args.unit)

    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE (raw): {mae_val:.6f} [{args.unit}]")
    if mae_mev is not None and mae_kcal is not None:
        print(f"MAE (meV): {mae_mev:.3f}")
        print(f"MAE (kcal/mol): {mae_kcal:.6f}")
    print(f"R2: {r2:.6f}")
    print(f"Inference total time (s): {inference_total_sec:.6f}")
    print(f"Latency per sample (ms): {inference_ms_per_sample:.6f}")
    print(f"Throughput (samples/s): {samples_per_sec:.6f}")
    print(f"Model params: {param_count:,}")
    print(f"Model size (MB): {model_size_mb:.4f}")
    if peak_gpu_mem_mb is not None:
        print(f"Peak GPU memory (MB): {peak_gpu_mem_mb:.4f}")

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out_df = pd.DataFrame({
            "idx_eval": out_indices[:len(y_pred_arr)],
            "idx_original_sdf_csv": out_original_indices[:len(y_pred_arr)],
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "abs_error": np.abs(y_true_arr - y_pred_arr),
        })
        out_df.to_csv(args.output, index=False)

        meta = {
            "model": args.model,
            "sdf": args.sdf,
            "csv": args.csv,
            "target": args.target,
            "split_file": args.split_file,
            "split": args.split,
            "unit": args.unit,
            "batch_size": args.batch_size,
            "num_samples": num_samples,
            "cutoff": cutoff,
            "num_rbf": num_rbf,
            "normalize_target": normalize_target,
            "per_atom_norm": per_atom_norm,
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

    if args.plot:
        plot_dir = os.path.dirname(args.plot)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
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
    p = argparse.ArgumentParser(description="Predict using ComplexPolarTransformer benchmark-ready")
    p.add_argument("--sdf", required=True, help="Path to qm9.sdf")
    p.add_argument("--csv", required=True, help="Path to CSV with targets aligned to SDF")
    p.add_argument("--target", default="u0_atom", help="Target column name in CSV")
    p.add_argument("--model", required=True, help="Path to model checkpoint")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda or cpu; auto if omitted")
    p.add_argument("--output", default="results/predictions.csv", help="Output CSV path")
    p.add_argument("--plot", default="results/pred_vs_real.png", help="Output plot path")
    p.add_argument("--show", action="store_true", help="Show plot interactively")
    p.add_argument("--dry-run", type=int, default=0, help="Stop after N samples for quick debug")
    p.add_argument("--split-file", default=None, help="Path to split_seed*.json from training")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--unit", default="kcal", choices=["hartree", "ev", "mev", "kcal"], help="Unidad del target en CSV")
    p.add_argument("--cutoff", type=float, default=None, help="Fallback cutoff if checkpoint has no hparams")
    p.add_argument("--force-first-target", action="store_true", help="Solo para checkpoints multi-target")
    p.add_argument("--allow-partial-load", action="store_true", help="Solo depuración; no usar para métricas oficiales")
    return p.parse_args()


if __name__ == "__main__":
    predict(parse_args())
