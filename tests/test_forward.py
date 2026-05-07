import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.complex_model_beta import ComplexPolarTransformerBeta


def fully_connected_edges(n_atoms: int):
    edges = []
    attrs = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            edges.append([i, j])
            attrs.append([1.0, 0.0, 0.0, 0.0])  # distancia real en Å + attrs auxiliares
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(attrs, dtype=torch.float32)


def run_test():
    in_dim = 5
    hidden_dim = 16
    out_dim = 1

    model = ComplexPolarTransformerBeta(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_hidden_layers=2,
        num_rbf=8,
        cutoff=3.0,
        dropout=0.1,
        use_residuals=True,
        use_layernorm=True,
    )
    model.eval()

    mol1 = torch.randn(3, in_dim)
    mol2 = torch.randn(2, in_dim)
    ei1, ea1 = fully_connected_edges(3)
    ei2, ea2 = fully_connected_edges(2)

    batch = {
        "atom_types": [mol1, mol2],
        "coords_spherical": [torch.randn(3, 3), torch.randn(2, 3)],
        "coords_cart": [torch.randn(3, 3), torch.randn(2, 3)],
        "edge_index": [ei1, ei2],
        "edge_attr": [ea1, ea2],
        "y": torch.tensor([0.0, 0.0]),
    }

    with torch.no_grad():
        out = model(batch)

    assert isinstance(out, torch.Tensor), "Output debe ser un Tensor"
    assert out.shape == (2, out_dim), f"Output shape esperado (2,{out_dim}), obtenido {out.shape}"

    # También validamos que el forward sea robusto sin aristas.
    batch_no_edges = dict(batch)
    batch_no_edges["edge_index"] = [None, None]
    batch_no_edges["edge_attr"] = [None, None]
    with torch.no_grad():
        out_no_edges = model(batch_no_edges)
    assert out_no_edges.shape == (2, out_dim)

    print("test_forward: OK — output shape:", out.shape)


if __name__ == "__main__":
    run_test()
