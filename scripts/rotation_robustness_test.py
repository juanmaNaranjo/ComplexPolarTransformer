"""
Experimento de robustez rotacional para PolCT v9.

Reevalúa el checkpoint ya entrenado de v9 (sin reentrenar) sobre el conjunto
de prueba aplicando a cada molécula una rotación SO(3) aleatoria e
independiente ANTES del centrado y de la conversión a coordenadas esféricas
que realiza QM9SDFDataset. Se compara contra un control idéntico sin rotar.

Como la energía de atomización es invariante ante rotaciones rígidas, un
modelo invariante debería producir exactamente las mismas predicciones.

Uso (desde la raíz del proyecto v9):
    python scripts/rotation_robustness_test.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import QM9SDFDataset  # noqa: E402
from predict_benchmark import (  # noqa: E402
    build_model_from_ckpt,
    collate_mol,
    load_checkpoint,
    load_split,
    tensor_stat_from_ckpt,
)


class RotatedQM9SDFDataset(QM9SDFDataset):
    """Igual a QM9SDFDataset, salvo que rota cada molécula con una rotación
    SO(3) uniforme determinada por (rotation_seed, índice original).
    Con rotation_seed=None se comporta exactamente como la clase original."""

    rotation_seed = None

    def rotation_for(self, idx):
        seed = int(self.rotation_seed) * 1_000_003 + int(self.original_indices[idx])
        return Rotation.random(random_state=seed).as_matrix()

    def __getitem__(self, idx):
        if self.rotation_seed is None:
            return super().__getitem__(idx)

        original = self.mols[idx]
        rotated = Chem.Mol(original)
        conf = rotated.GetConformer()
        pos = conf.GetPositions() @ self.rotation_for(idx).T
        for i, (x, y, z) in enumerate(pos):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

        self.mols[idx] = rotated
        try:
            return super().__getitem__(idx)
        finally:
            self.mols[idx] = original


def run_pass(model, dataset, ckpt, device, batch_size, num_workers, dry_run):
    y_mean = tensor_stat_from_ckpt(ckpt, "y_mean", [0.0], device)
    y_std = tensor_stat_from_ckpt(ckpt, "y_std", [1.0], device)
    per_atom_norm = bool(ckpt.get("per_atom_norm", False))

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_mol, num_workers=num_workers)
    y_true, y_pred, orig_idx, n_edges = [], [], [], []
    t0 = time.perf_counter()
    for batch in tqdm(dl, desc="Predict"):
        for key in ("atom_types", "coords_spherical", "coords_cart", "edge_index", "edge_attr"):
            batch[key] = [t.to(device) for t in batch[key]]
        with torch.no_grad():
            preds = torch.as_tensor(model(batch)).float().view(-1, 1)
        preds = preds * y_std + y_mean
        if per_atom_norm:
            preds = preds * batch["num_atoms"].to(device).view(-1, 1).clamp_min(1.0)
        y_true.append(batch["y"].numpy().reshape(-1))
        y_pred.append(preds.cpu().numpy().reshape(-1))
        orig_idx.append(batch["original_idx"].numpy())
        n_edges.append(batch["num_edges"].numpy())
        if dry_run and sum(len(a) for a in y_true) >= dry_run:
            break
    elapsed = time.perf_counter() - t0
    return (np.concatenate(y_true), np.concatenate(y_pred),
            np.concatenate(orig_idx), np.concatenate(n_edges), elapsed)


def metrics(y_true, y_pred):
    return {
        "mae_kcal_mol": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kcal_mol": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    p = argparse.ArgumentParser(description="Robustez rotacional de PolCT v9")
    p.add_argument("--sdf", default="data/qm9.sdf")
    p.add_argument("--csv", default="data/qm9.csv")
    p.add_argument("--target", default="u0_atom")
    p.add_argument("--model", default="checkpoints_v9_angular_modrelu/best_model.pt")
    p.add_argument("--split-file", default="logs_v9_angular_modrelu/split_seed42.json")
    p.add_argument("--rotation-seeds", type=int, nargs="+", default=[123, 456, 789])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dry-run", type=int, default=0, help="Evaluar solo N moléculas (depuración)")
    p.add_argument("--out-dir", default="results/rotation_robustness")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt, state_dict = load_checkpoint(args.model, device)
    model, model_cfg = build_model_from_ckpt(ckpt, state_dict)
    model.to(device).eval()

    dataset = RotatedQM9SDFDataset(args.sdf, args.csv, target_col=args.target,
                                   max_radius=float(model_cfg["cutoff"]))
    test_idx, _ = load_split(args.split_file, "test")
    test_set = Subset(dataset, test_idx)
    os.makedirs(args.out_dir, exist_ok=True)

    runs = {}
    dataset.rotation_seed = None
    yt, yc, oi, ne_ctrl, sec = run_pass(model, test_set, ckpt, device,
                                        args.batch_size, args.num_workers, args.dry_run)
    runs["control"] = {**metrics(yt, yc), "elapsed_sec": sec, "n": int(len(yt))}
    table = pd.DataFrame({"idx_original_sdf_csv": oi, "y_true": yt, "y_pred_control": yc})

    for seed in args.rotation_seeds:
        dataset.rotation_seed = seed
        yt_r, yr, oi_r, ne_rot, sec = run_pass(model, test_set, ckpt, device,
                                               args.batch_size, args.num_workers, args.dry_run)
        assert np.array_equal(oi_r, oi) and np.allclose(yt_r, yt)
        shift = np.abs(yr - yc)
        runs[f"rot_seed_{seed}"] = {
            **metrics(yt, yr),
            "elapsed_sec": sec,
            "n": int(len(yt)),
            # Sanidad: una rotación rígida no debe cambiar el grafo (aristas por distancia).
            "edges_identical_to_control": bool(np.array_equal(ne_rot, ne_ctrl)),
            "pred_shift_mean_kcal_mol": float(shift.mean()),
            "pred_shift_median_kcal_mol": float(np.median(shift)),
            "pred_shift_p99_kcal_mol": float(np.percentile(shift, 99)),
            "pred_shift_max_kcal_mol": float(shift.max()),
        }
        table[f"y_pred_rot_{seed}"] = yr

    rot = [v for k, v in runs.items() if k.startswith("rot_")]
    mae_rot = np.array([r["mae_kcal_mol"] for r in rot])
    summary = {
        "mae_control": runs["control"]["mae_kcal_mol"],
        "mae_rot_mean": float(mae_rot.mean()),
        "mae_rot_std": float(mae_rot.std(ddof=1)) if len(mae_rot) > 1 else 0.0,
        "mae_ratio_mean": float(mae_rot.mean() / runs["control"]["mae_kcal_mol"]),
        "pred_shift_mean_over_seeds": float(np.mean([r["pred_shift_mean_kcal_mol"] for r in rot])),
    }
    out = {
        "model": args.model,
        "split_file": args.split_file,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "rotation": "scipy Rotation.random (uniforme en SO(3)); semilla = seed*1000003 + idx_original",
        "applied_before": "centrado en el centroide y conversión a coordenadas esféricas (QM9SDFDataset)",
        "device": device,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
        "runs": runs,
        "summary": summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    with open(os.path.join(args.out_dir, "rotation_robustness_full.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    table.to_csv(os.path.join(args.out_dir, "rotation_predictions.csv"), index=False)
    print(json.dumps({"runs": runs, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
