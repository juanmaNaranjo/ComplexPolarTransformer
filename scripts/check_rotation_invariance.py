"""
Verifica sensibilidad a rotación global para un checkpoint.

Uso:
python scripts/check_rotation_invariance.py \
  --sdf data/qm9.sdf \
  --csv data/qm9.csv \
  --target u0_atom \
  --model checkpoints_v11_invariant_angular/best_model.pt \
  --index 0

En geometry_mode='invariant', la diferencia debe ser muy cercana a cero.
En geometry_mode='polar', normalmente puede cambiar porque theta/phi dependen del eje global.
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dataset import QM9SDFDataset
from predict_benchmark import load_checkpoint, build_model_from_ckpt


def random_rotation(device="cpu"):
    a = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(a)
    # Asegurar determinante positivo para una rotación propia.
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def make_single_batch(sample, device):
    return {
        "atom_types": [sample["atom_types"].to(device)],
        "coords_spherical": [sample["coords_spherical"].to(device)],
        "coords_cart": [sample["coords_cart"].to(device)],
        "edge_index": [sample["edge_index"].to(device)],
        "edge_attr": [sample["edge_attr"].to(device)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="u0_atom")
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, state_dict = load_checkpoint(args.model, device)
    model, model_cfg = build_model_from_ckpt(ckpt, state_dict)
    model.to(device).eval()

    cutoff = float(model_cfg.get("cutoff", 7.0))
    dataset = QM9SDFDataset(args.sdf, args.csv, target_col=args.target, max_radius=cutoff)
    sample = dataset[args.index]

    batch_1 = make_single_batch(sample, device)

    # Rotar coords_cart. No se cambia edge_attr porque las distancias son invariantes.
    # coords_spherical se deja igual: en geometry_mode='invariant' no se usa para el embedding.
    rot = random_rotation(device=device)
    sample_rot = dict(sample)
    sample_rot["coords_cart"] = sample["coords_cart"].to(device) @ rot.T
    batch_2 = make_single_batch(sample_rot, device)

    with torch.no_grad():
        pred_1 = model(batch_1).view(-1)[0]
        pred_2 = model(batch_2).view(-1)[0]

    diff = torch.abs(pred_1 - pred_2).item()
    print("geometry_mode:", model_cfg.get("geometry_mode", "polar"))
    print("pred original:", float(pred_1.detach().cpu()))
    print("pred rotada:  ", float(pred_2.detach().cpu()))
    print("abs diff:     ", diff)


if __name__ == "__main__":
    main()
