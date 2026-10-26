"""
Pruebas complementarias de PolCT v9 sobre el conjunto de prueba (sin reentrenar):

  1. Invariancia a la permutación de átomos: se reordena aleatoriamente la
     numeración de los átomos de cada molécula. La energía no depende del
     orden, así que un modelo bien construido debe dar la misma predicción.
  2. Invariancia a la reflexión (x -> -x): la energía de atomización es
     invariante ante una inversión especular de la geometría.
  3. Latencia de la red sola: se precalculan los grafos y se mide únicamente
     el paso hacia adelante en GPU, para separar el costo de la red del de la
     construcción del grafo en Python.

Uso (desde la raíz del proyecto v9):
    python scripts/symmetry_latency_tests.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
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
from scripts.rotation_robustness_test import metrics  # noqa: E402

GRAPH_KEYS = ("atom_types", "coords_spherical", "coords_cart", "edge_index", "edge_attr")


class TransformedQM9SDFDataset(QM9SDFDataset):
    """Aplica 'permute' o 'reflect' a una copia en memoria de cada molécula
    antes del centrado y la conversión a esféricas. transform=None = original."""

    transform = None
    seed = 2024

    def _transformed(self, mol, idx):
        if self.transform == "permute":
            rng = np.random.default_rng(self.seed * 1_000_003 + int(self.original_indices[idx]))
            order = [int(i) for i in rng.permutation(mol.GetNumAtoms())]
            return Chem.RenumberAtoms(mol, order)
        if self.transform == "reflect":
            m = Chem.Mol(mol)
            conf = m.GetConformer()
            for i, (x, y, z) in enumerate(conf.GetPositions()):
                conf.SetAtomPosition(i, Point3D(-float(x), float(y), float(z)))
            return m
        raise ValueError(self.transform)

    def __getitem__(self, idx):
        if self.transform is None:
            return super().__getitem__(idx)
        original = self.mols[idx]
        self.mols[idx] = self._transformed(original, idx)
        try:
            return super().__getitem__(idx)
        finally:
            self.mols[idx] = original


def to_device(batch, device):
    for key in GRAPH_KEYS:
        batch[key] = [t.to(device) for t in batch[key]]
    return batch


def denorm(preds, batch, ckpt, device):
    y_mean = tensor_stat_from_ckpt(ckpt, "y_mean", [0.0], device)
    y_std = tensor_stat_from_ckpt(ckpt, "y_std", [1.0], device)
    preds = torch.as_tensor(preds).float().view(-1, 1) * y_std + y_mean
    if ckpt.get("per_atom_norm", False):
        preds = preds * batch["num_atoms"].to(device).view(-1, 1).clamp_min(1.0)
    return preds.cpu().numpy().reshape(-1)


def run_pass(model, dataset, ckpt, device, batch_size, keep_batches=False):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mol)
    y_true, y_pred, kept = [], [], []
    t0 = time.perf_counter()
    for batch in tqdm(dl, desc="Predict"):
        batch = to_device(batch, device)
        with torch.no_grad():
            y_pred.append(denorm(model(batch), batch, ckpt, device))
        y_true.append(batch["y"].numpy().reshape(-1))
        if keep_batches:
            kept.append(batch)
    elapsed = time.perf_counter() - t0
    return np.concatenate(y_true), np.concatenate(y_pred), elapsed, kept


def forward_only_latency(model, batches, device, repeats):
    n = sum(len(b["atom_types"]) for b in batches)
    with torch.no_grad():
        for b in batches[:20]:  # calentamiento
            model(b)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for b in batches:
                model(b)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    return {"n_molecules": n, "repeats": repeats, "total_sec_median": float(np.median(times)),
            "ms_per_molecule_median": float(np.median(times) / n * 1000),
            "ms_per_molecule_min": float(times.min() / n * 1000),
            "molecules_per_sec_median": float(n / np.median(times))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", default="data/qm9.sdf")
    p.add_argument("--csv", default="data/qm9.csv")
    p.add_argument("--target", default="u0_atom")
    p.add_argument("--model", default="checkpoints_v9_angular_modrelu/best_model.pt")
    p.add_argument("--split-file", default="logs_v9_angular_modrelu/split_seed42.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--latency-repeats", type=int, default=3)
    p.add_argument("--dry-run", type=int, default=0)
    p.add_argument("--out-dir", default="results/symmetry_latency")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt, state_dict = load_checkpoint(args.model, device)
    model, model_cfg = build_model_from_ckpt(ckpt, state_dict)
    model.to(device).eval()

    dataset = TransformedQM9SDFDataset(args.sdf, args.csv, target_col=args.target,
                                       max_radius=float(model_cfg["cutoff"]))
    test_idx, _ = load_split(args.split_file, "test")
    if args.dry_run:
        test_idx = test_idx[: args.dry_run]
    test_set = Subset(dataset, test_idx)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset.transform = None
    yt, yc, sec_ctrl, batches = run_pass(model, test_set, ckpt, device, args.batch_size,
                                         keep_batches=True)
    out = {"model": args.model, "checkpoint_epoch": int(ckpt.get("epoch", -1)), "device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "batch_size": args.batch_size, "n": int(len(yt)),
           "control": {**metrics(yt, yc), "end_to_end_sec": sec_ctrl,
                       "end_to_end_ms_per_molecule": sec_ctrl / len(yt) * 1000}}

    out["latency_forward_only"] = forward_only_latency(model, batches, device, args.latency_repeats)
    out["latency_forward_only"]["share_of_end_to_end_pct"] = float(
        out["latency_forward_only"]["ms_per_molecule_median"]
        / out["control"]["end_to_end_ms_per_molecule"] * 100)
    del batches

    for name in ("permute", "reflect"):
        dataset.transform = name
        yt_t, yp, sec, _ = run_pass(model, test_set, ckpt, device, args.batch_size)
        assert np.allclose(yt_t, yt)
        shift = np.abs(yp - yc)
        out[name] = {**metrics(yt, yp), "elapsed_sec": sec,
                     "mae_change_pct": float((metrics(yt, yp)["mae_kcal_mol"]
                                              / out["control"]["mae_kcal_mol"] - 1) * 100),
                     "pred_shift_mean_kcal_mol": float(shift.mean()),
                     "pred_shift_median_kcal_mol": float(np.median(shift)),
                     "pred_shift_p99_kcal_mol": float(np.percentile(shift, 99)),
                     "pred_shift_max_kcal_mol": float(shift.max())}
        np.save(os.path.join(args.out_dir, f"pred_{name}.npy"), yp)

    out["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(os.path.join(args.out_dir, "symmetry_latency.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
