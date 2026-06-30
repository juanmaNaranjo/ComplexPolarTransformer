import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.complex_model_beta import ComplexPolarTransformerBeta

# in_dim=12: 5 one-hot atómico + 3 hibridación + arom + ring + charge + Hs
_IN_DIM = 12


def fully_connected_edges(n_atoms: int):
    """[E, 2] → [2, E] edge_index y [E, 1] edge_attr (distancia ~1.0 Å)."""
    edges = []
    attrs = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            edges.append([i, j])
            attrs.append([1.0])  # columna 0 = distancia real en Å; RBFExpansion usa solo col 0
    return (
        torch.tensor(edges, dtype=torch.long).t().contiguous(),
        torch.tensor(attrs, dtype=torch.float32),
    )


def run_test():
    hidden_dim = 16
    out_dim    = 1

    model = ComplexPolarTransformerBeta(
        in_dim=_IN_DIM,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_hidden_layers=2,
        num_heads=4,
        num_rbf=8,
        cutoff=3.0,
        dropout=0.1,
        use_residuals=True,
        use_layernorm=True,
        activation="modrelu",
        modrelu_init_bias=-0.1,
        use_angular=True,
        num_angle_basis=6,
        angular_scale_init=0.1,
    )
    model.eval()

    mol1 = torch.randn(3, _IN_DIM)
    mol2 = torch.randn(2, _IN_DIM)
    ei1, ea1 = fully_connected_edges(3)
    ei2, ea2 = fully_connected_edges(2)

    batch = {
        "atom_types":       [mol1, mol2],
        "coords_spherical": [torch.randn(3, 3), torch.randn(2, 3)],
        "coords_cart":      [torch.randn(3, 3), torch.randn(2, 3)],
        "edge_index":       [ei1, ei2],
        "edge_attr":        [ea1, ea2],
        "y": torch.tensor([0.0, 0.0]),
    }

    with torch.no_grad():
        out = model(batch)

    assert isinstance(out, torch.Tensor), "Output debe ser un Tensor"
    assert out.shape == (2, out_dim), \
        f"Output shape esperado (2,{out_dim}), obtenido {out.shape}"

    # Forward robusto sin aristas (moléculas aisladas / batch edge-free)
    batch_no_edges = dict(batch)
    batch_no_edges["edge_index"] = [None, None]
    batch_no_edges["edge_attr"]  = [None, None]
    with torch.no_grad():
        out_no_edges = model(batch_no_edges)
    assert out_no_edges.shape == (2, out_dim)

    print("test_forward: OK — output shape:", out.shape)


if __name__ == "__main__":
    run_test()
