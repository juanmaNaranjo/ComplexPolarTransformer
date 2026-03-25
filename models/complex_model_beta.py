import torch
import torch.nn as nn
from .complex_layers import (
    ComplexEmbedding,
    ComplexPolarAttention,
    ComplexLayerNorm,
    ComplexFFN,
    RealProjection,
)


class ComplexPolarTransformerBeta(nn.Module):
    """
    Transformer complejo-polar para predicción de propiedades moleculares.
    Versión completa: embedding → N × (atención + LN + FFN + LN) → proyección real
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_hidden_layers=1,
        dropout=0.0,
        use_residuals=False,
        use_layernorm=False,
        edge_dim=4,
        **kwargs
    ):
        super().__init__()

        self.use_residuals = use_residuals
        self.use_layernorm = use_layernorm
        self.num_hidden_layers = num_hidden_layers

        # Entrada = atom_types (in_dim) + coordenadas esféricas (3)
        self.input_dim = in_dim + 3

        # Embedding complejo
        self.embedding = ComplexEmbedding(
            in_dim=self.input_dim,
            hidden_dim=hidden_dim
        )

        # Capas apiladas: atención + FFN
        self.attn_layers = nn.ModuleList([
            ComplexPolarAttention(hidden_dim=hidden_dim, edge_dim=edge_dim)
            for _ in range(num_hidden_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            ComplexFFN(hidden_dim=hidden_dim)
            for _ in range(num_hidden_layers)
        ])

        if use_layernorm:
            self.ln_attn = nn.ModuleList([
                ComplexLayerNorm(hidden_dim)
                for _ in range(num_hidden_layers)
            ])
            self.ln_ffn = nn.ModuleList([
                ComplexLayerNorm(hidden_dim)
                for _ in range(num_hidden_layers)
            ])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Proyección final a valores reales
        self.out_proj = RealProjection(hidden_dim, out_dim)

    def forward(self, batch):
        atom_feats = batch["atom_types"]
        coords_sph = batch["coords_spherical"]

        # edge_index y edge_attr opcionales
        edge_indices = batch.get("edge_index", [None] * len(atom_feats))
        edge_attrs   = batch.get("edge_attr",  [None] * len(atom_feats))

        mol_outputs = []

        for feats, sph, eidx, eattr in zip(
            atom_feats, coords_sph, edge_indices, edge_attrs
        ):
            # Concatenar features químicas + geometría polar
            x = torch.cat([feats, sph], dim=-1)  # [N, in_dim+3]

            # Embedding complejo
            z = self.embedding(x)

            # Bloques apilados
            for i in range(self.num_hidden_layers):

                # --- Atención ---
                z_attn = self.attn_layers[i](z, edge_index=eidx, edge_attr=eattr)

                if self.use_residuals:
                    z_attn = z_attn.add(z)

                if self.use_layernorm:
                    z_attn = self.ln_attn[i](z_attn)

                # --- FFN ---
                z_ffn = self.ffn_layers[i](z_attn)

                if self.use_residuals:
                    z_ffn = z_ffn.add(z_attn)

                if self.use_layernorm:
                    z_ffn = self.ln_ffn[i](z_ffn)

                z = z_ffn

            # Proyección real por átomo
            atom_out = self.out_proj(z)   # [N, out_dim]

            if self.dropout:
                atom_out = self.dropout(atom_out)

            # Pooling molecular
            mol_outputs.append(atom_out.sum(dim=0))

        return torch.stack(mol_outputs)