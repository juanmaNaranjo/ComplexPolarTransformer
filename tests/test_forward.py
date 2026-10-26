import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.complex_model_beta import ComplexPolarTransformerBeta


def make_batch():
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long)
    edge_attr = torch.tensor([
        [1.1, 0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    return {
        "atom_types": [torch.randn(3, 5), torch.randn(4, 5)],
        "coords_spherical": [torch.randn(3, 3), torch.randn(4, 3)],
        "coords_cart": [torch.randn(3, 3), torch.randn(4, 3)],
        "edge_index": [edge_index, edge_index],
        "edge_attr": [edge_attr, edge_attr],
    }


def run_case(geometry_mode):
    model = ComplexPolarTransformerBeta(
        in_dim=5,
        hidden_dim=32,
        out_dim=1,
        num_hidden_layers=2,
        num_rbf=16,
        cutoff=7.0,
        activation="modrelu",
        use_angular=True,
        num_angle_basis=8,
        geometry_mode=geometry_mode,
    )
    out = model(make_batch())
    assert out.shape == (2, 1)
    print(f"test_forward {geometry_mode}: OK — output shape: {out.shape}")


if __name__ == "__main__":
    run_case("polar")
    run_case("invariant")
