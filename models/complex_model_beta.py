from .complex_layers import (
    RBFExpansion,
    ComplexEmbedding,
    ComplexPolarAttention,
    ComplexMessagePassing,
    RealProjection,
)
import torch
import torch.nn as nn
import math


class ComplexPolarTransformerBeta(nn.Module):
    """
    Transformer complejo-polar v6 — arquitectura con RBF y MP.

    Componentes por capa (se repite num_hidden_layers veces):

        rbf_ij = RBFExpansion(d_ij)          ← NUEVO: distancia → 50 gaussianas
        z_attn = ComplexPolarAttention(z, rbf)  ← usa rbf en vez de edge_attr raw
        z_msg  = ComplexMessagePassing(z_attn, rbf)  ← MP con representación rica
        z      = residual + LayerNorm

    RBFExpansion (Schütt 2018 [14]):
        Convierte la distancia escalar d_ij en 50 gaussianas centradas
        uniformemente en [0, 5Å], moduladas por un cosine cutoff físico.
        Permite aprender cualquier función radial suave — clave para
        modelar potenciales de enlace, van der Waals, y electrostática.

    ComplexMessagePassing con RBF:
        Los mensajes entre átomos vecinos ahora están ponderados por
        la representación RBF en vez de los 4 valores crudos de edge_attr.
        Esto da al modelo la capacidad de aprender perfiles de interacción
        físicamente realistas en el dominio complejo.

    Sin cambios respecto a la base:
        - Atención compleja (magnitud × magnitud, fase - fase)
        - Residuals + LayerNorm entre capas
        - Pooling mean + sum
        - out_head simple (lección de v4: sin LN en el head)

    Args:
        in_dim:            dimensión atom_types (5)
        hidden_dim:        dimensión latente (256)
        out_dim:           propiedades (1)
        num_hidden_layers: bloques (3)
        num_rbf:           gaussianas RBF (50)
        cutoff:            cutoff físico en Å (5.0)
        edge_dim:          dimensión original de edge_attr (4, para compatibilidad)
        dropout:           dropout (0.1)
        use_residuals:     residual entre bloques (True)
        use_layernorm:     LayerNorm entre bloques (True)
    """

    def __init__(
        self,
        in_dim:            int   = 5,
        hidden_dim:        int   = 256,
        out_dim:           int   = 1,
        num_hidden_layers: int   = 3,
        num_rbf:           int   = 50,
        cutoff:            float = 5.0,
        edge_dim:          int   = 4,     # ignorado — se usa num_rbf
        dropout:           float = 0.1,
        use_residuals:     bool  = True,
        use_layernorm:     bool  = True,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim    = hidden_dim
        self.use_residuals = use_residuals
        self.use_layernorm = use_layernorm

        self.input_dim = in_dim + 3   # atom_types + (r, θ, φ)

        # RBF: convierte distancia → 50 gaussianas (sin params)
        self.rbf = RBFExpansion(num_rbf=num_rbf, cutoff=cutoff)

        # Embedding inicial
        self.embedding = ComplexEmbedding(self.input_dim, hidden_dim)

        # Bloques: atención + MP (ambos reciben RBF)
        self.attn_layers = nn.ModuleList([
            ComplexPolarAttention(hidden_dim=hidden_dim, edge_dim=num_rbf)
            for _ in range(num_hidden_layers)
        ])

        self.mp_layers = nn.ModuleList([
            ComplexMessagePassing(hidden_dim=hidden_dim, edge_dim=num_rbf)
            for _ in range(num_hidden_layers)
        ])

        # LayerNorm entre bloques
        if use_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_hidden_layers)
            ])
        else:
            self.layer_norms = None

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Proyección por átomo
        self.out_proj = RealProjection(hidden_dim, hidden_dim)

        # out_head sin LayerNorm (lección de v4)
        pool_dim = hidden_dim * 2
        self.out_head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        atom_feats = batch["atom_types"]
        coords_sph = batch["coords_spherical"]
        edge_index = batch.get("edge_index", [None] * len(atom_feats))
        edge_attr  = batch.get("edge_attr",  [None] * len(atom_feats))

        mol_outputs = []

        for feats, sph, ei, ea in zip(atom_feats, coords_sph, edge_index, edge_attr):

            x = torch.cat([feats.float(), sph.float()], dim=-1)
            z = self.embedding(x)

            # Calcular RBF una sola vez por molécula
            rbf = self.rbf(ea.float()) if ea is not None else None

            for layer_idx in range(len(self.attn_layers)):
                # Atención compleja con bias RBF
                z_new = self.attn_layers[layer_idx](z, edge_index=ei, rbf=rbf)

                # Message passing con RBF
                if ei is not None and rbf is not None:
                    z_new = self.mp_layers[layer_idx](z_new, ei, rbf)

                # Residual
                if self.use_residuals:
                    z_new.magnitude = z_new.magnitude + z.magnitude
                    z_new.phase     = z_new.phase     + z.phase

                # LayerNorm
                if self.use_layernorm and self.layer_norms is not None:
                    z_new.magnitude = self.layer_norms[layer_idx](z_new.magnitude)

                z_new.magnitude = self.dropout(z_new.magnitude)
                z = z_new

            # Pooling
            atom_repr = self.out_proj(z)
            mol_repr  = torch.cat([atom_repr.mean(dim=0), atom_repr.sum(dim=0)], dim=-1)
            mol_outputs.append(mol_repr)

        return self.out_head(torch.stack(mol_outputs))