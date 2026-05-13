import torch
import torch.nn as nn

from .complex_layers import (
    RBFExpansion,
    ComplexEmbedding,
    ComplexPolarAttention,
    ComplexMessagePassing,
    RealProjection,
)


class ComplexPolarTransformerBeta(nn.Module):
    """
    ComplexPolarTransformer — v8 (mejoras MAE).

    Cambios respecto a v7:
    - ModReLU como activación polar en ComplexMessagePassing y ComplexEmbedding.
    - Atención sparse O(E) en ComplexPolarAttention (no O(N²)).
    - RBFExpansion incluye features angulares sin(Δθ)/cos(Δθ)/sin(Δφ)/cos(Δφ).
      edge_dim interno = num_rbf + 4.
    - input_dim = in_dim + 3 (soporta in_dim=9 para features atómicas enriquecidas).
    - LayerNorm independiente en magnitud Y fase dentro de cada capa.
    - 6 capas ocultas y hidden_dim=384 por defecto.
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 384,
        out_dim: int = 1,
        num_hidden_layers: int = 6,
        num_rbf: int = 150,
        cutoff: float = 7.0,
        edge_dim: int = 4,        # conservado por compatibilidad; internamente se usa num_rbf+4
        dropout: float = 0.05,
        use_residuals: bool = True,
        use_layernorm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.use_residuals = bool(use_residuals)
        self.use_layernorm = bool(use_layernorm)

        # in_dim (features atómicas) + 3 coordenadas esféricas (r, θ, φ)
        self.input_dim = int(in_dim) + 3

        # edge_dim real: RBF radiales + 4 features angulares
        self._edge_dim = self.num_rbf + 4

        self.rbf = RBFExpansion(num_rbf=self.num_rbf, cutoff=self.cutoff)
        self.embedding = ComplexEmbedding(self.input_dim, self.hidden_dim)

        self.attn_layers = nn.ModuleList([
            ComplexPolarAttention(
                hidden_dim=self.hidden_dim,
                edge_dim=self._edge_dim,
            )
            for _ in range(self.num_hidden_layers)
        ])

        self.mp_layers = nn.ModuleList([
            ComplexMessagePassing(
                hidden_dim=self.hidden_dim,
                edge_dim=self._edge_dim,
            )
            for _ in range(self.num_hidden_layers)
        ])

        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.hidden_dim)
                for _ in range(self.num_hidden_layers)
            ])
        else:
            self.layer_norms = None

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.out_proj = RealProjection(self.hidden_dim, self.hidden_dim)

        pool_dim = self.hidden_dim * 2
        self.out_head = nn.Sequential(
            nn.Linear(pool_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, out_dim),
        )

    @staticmethod
    def _as_list_or_default(value, n_items):
        if value is None:
            return [None] * n_items
        return value

    @staticmethod
    def _has_edges(edge_index, edge_attr):
        return (
            edge_index is not None
            and edge_attr is not None
            and isinstance(edge_index, torch.Tensor)
            and isinstance(edge_attr, torch.Tensor)
            and edge_index.numel() > 0
            and edge_attr.numel() > 0
        )

    def forward(self, batch: dict) -> torch.Tensor:
        atom_feats = batch["atom_types"]
        coords_sph = batch["coords_spherical"]

        edge_index_list = self._as_list_or_default(batch.get("edge_index"), len(atom_feats))
        edge_attr_list  = self._as_list_or_default(batch.get("edge_attr"),  len(atom_feats))

        mol_outputs = []

        for feats, sph, ei, ea in zip(atom_feats, coords_sph, edge_index_list, edge_attr_list):
            x = torch.cat([feats.float(), sph.float()], dim=-1)  # [N, in_dim+3]
            z = self.embedding(x)

            if self._has_edges(ei, ea):
                rbf = self.rbf(ea.float())   # [E, num_rbf+4]
            else:
                ei = None
                rbf = None

            for layer_idx in range(self.num_hidden_layers):
                # Atención sparse compleja-polar
                z_new = self.attn_layers[layer_idx](z, edge_index=ei, rbf=rbf)

                # Message passing con modReLU
                if ei is not None and rbf is not None:
                    z_new = self.mp_layers[layer_idx](z_new, ei, rbf)

                # Conexiones residuales
                if self.use_residuals:
                    z_new.magnitude = z_new.magnitude + z.magnitude
                    z_new.phase     = z_new.phase     + z.phase

                # LayerNorm en magnitud
                if self.use_layernorm and self.layer_norms is not None:
                    z_new.magnitude = self.layer_norms[layer_idx](z_new.magnitude)

                z_new.magnitude = self.dropout(z_new.magnitude)
                z = z_new

            # Pooling: mean + sum sobre átomos → representación molecular
            atom_repr = self.out_proj(z)                                    # [N, hidden_dim]
            mol_repr  = torch.cat([atom_repr.mean(dim=0),
                                   atom_repr.sum(dim=0)], dim=-1)           # [2*hidden_dim]
            mol_outputs.append(mol_repr)

        return self.out_head(torch.stack(mol_outputs))                      # [B, out_dim]
