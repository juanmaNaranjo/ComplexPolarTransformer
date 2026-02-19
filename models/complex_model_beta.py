from .complex_layers import ComplexEmbedding, ComplexPolarAttention, RealProjection
import torch
import torch.nn as nn


class ComplexPolarTransformerBeta(nn.Module):
    """
    Modelo complejo-polar (versión beta) para predicción de propiedades moleculares.

    - Integra atom_types + coordenadas polares (r, θ, φ)
    - Usa representación compleja (magnitud + fase)
    - Incluye una interacción compleja tipo atención (simplificada)
    - Pooling por suma a nivel molecular

    Esta versión es un primer modelo defendible para tesis.
    """

    def __init__(
        self,
        in_dim,        # dimensión de atom_types (ej: 5)
        hidden_dim,    # dimensión latente
        out_dim,       # dimensión de salida (ej: 1)
        **kwargs       # compatibilidad con YAML (dropout, etc.)
    ):
        super().__init__()

        # Entrada total = atom_types + (r, θ, φ)
        self.input_dim = in_dim + 3

        # Embedding complejo (magnitud + fase)
        self.embedding = ComplexEmbedding(
            in_dim=self.input_dim,
            hidden_dim=hidden_dim
        )

        # Interacción compleja (placeholder de atención)
        self.attn = ComplexPolarAttention()


        # Proyección final a valores reales
        self.out_proj = RealProjection(hidden_dim, out_dim)

    def forward(self, batch):
        """
        batch contiene:
            - atom_types: List[Tensor] -> [(n_atoms_i, in_dim)]
            - coords_spherical: List[Tensor] -> [(n_atoms_i, 3)]
        """

        atom_feats = batch["atom_types"]
        coords_sph = batch["coords_spherical"]

        mol_outputs = []

        for feats, sph in zip(atom_feats, coords_sph):
            # feats: [n_atoms, in_dim]
            # sph:   [n_atoms, 3] -> (r, θ, φ)

            # Concatenar información química + geométrica
            x = torch.cat([feats, sph], dim=-1)

            # Embedding complejo
            z = self.embedding(x)  # ComplexTensor

            # Interacción compleja entre átomos
            z = self.attn(z)

            # Proyección a salida real por átomo
            atom_out = self.out_proj(z)  # [n_atoms, out_dim]

            # Pooling molecular (suma)
            mol_out = atom_out.sum(dim=0)
            mol_outputs.append(mol_out)

        return torch.stack(mol_outputs)
