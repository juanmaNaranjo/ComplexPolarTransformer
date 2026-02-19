import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.complex_model_beta import ComplexPolarTransformerBeta


def run_test():
    # Config
    in_dim = 5
    hidden_dim = 16
    out_dim = 1

    # Modelo con profundidad y regularización
    model = ComplexPolarTransformerBeta(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                        num_hidden_layers=4, dropout=0.1, use_residuals=True, use_layernorm=True)
    model.eval()

    # Crear batch sintético: dos moléculas, una con 3 átomos y otra con 2 átomos
    mol1 = torch.randn(3, in_dim)
    mol2 = torch.randn(2, in_dim)
    batch = {
        "atom_types": [mol1, mol2],
        "coords_spherical": [torch.randn(3, 3), torch.randn(2, 3)],
        "coords_cart": [torch.randn(3, 3), torch.randn(2, 3)],
        "edge_index": None,
        "edge_attr": None,
        "y": torch.tensor([0.0, 0.0])
    }

    with torch.no_grad():
        out = model(batch)

    assert isinstance(out, torch.Tensor), "Output debe ser un Tensor"
    assert out.shape == (2, out_dim), f"Output shape esperado (2,{out_dim}), obtenido {out.shape}"
    print("test_forward: OK — output shape:", out.shape)


if __name__ == '__main__':
    run_test()
