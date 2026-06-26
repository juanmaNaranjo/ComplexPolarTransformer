import torch
import torch.nn as nn

from .complex_layers import (
    RBFExpansion,
    ComplexEmbedding,
    ComplexPolarAttention,
    ComplexMessagePassing,
    ShiftedSoftplus,
    ModReLU,
)


class ComplexPolarTransformerBeta(nn.Module):
    """
    ComplexPolarTransformer benchmark-ready.

    Versión corregida para v7:
    - RBFExpansion usa distancias reales en Å desde edge_attr[:, 0].
    - cutoff del modelo debe coincidir con max_radius del dataset.
    - forward robusto ante edge_index/edge_attr None o tensores vacíos.
    """

    def __init__(
        self,
        in_dim: int = 5,
        hidden_dim: int = 256,
        out_dim: int = 1,
        num_hidden_layers: int = 3,
        num_rbf: int = 50,
        cutoff: float = 5.0,
        edge_dim: int = 4,  # conservado por compatibilidad; internamente se usa num_rbf
        dropout: float = 0.1,
        use_residuals: bool = True,
        use_layernorm: bool = True,
        activation: str = "modrelu",
        modrelu_init_bias: float = -0.1,
        modrelu_eps: float = 1e-8,
        use_angular: bool = False,
        num_angle_basis: int = 16,
        angular_scale_init: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)
        self.use_residuals = bool(use_residuals)
        self.use_layernorm = bool(use_layernorm)
        self.activation_name = str(activation).lower().strip()
        self.use_angular = bool(use_angular)
        self.num_angle_basis = int(num_angle_basis)
        self.angular_scale_init = float(angular_scale_init)
        self.input_dim = int(in_dim) + 3  # atom_types + coordenadas esféricas (r, theta, phi)

        self.rbf = RBFExpansion(num_rbf=self.num_rbf, cutoff=self.cutoff)
        self.embedding = ComplexEmbedding(self.input_dim, self.hidden_dim)

        if self.activation_name == "modrelu":
            self.input_activation = ModReLU(
                self.hidden_dim,
                init_bias=float(modrelu_init_bias),
                eps=float(modrelu_eps),
            )
            self.complex_activations = nn.ModuleList([
                ModReLU(
                    self.hidden_dim,
                    init_bias=float(modrelu_init_bias),
                    eps=float(modrelu_eps),
                )
                for _ in range(self.num_hidden_layers)
            ])
        elif self.activation_name in {"identity", "none", "linear"}:
            self.input_activation = nn.Identity()
            self.complex_activations = nn.ModuleList([nn.Identity() for _ in range(self.num_hidden_layers)])
        else:
            raise ValueError(
                f"activation no soportada: {activation}. Usa 'modrelu' o 'identity'."
            )

        self.attn_layers = nn.ModuleList([
            ComplexPolarAttention(hidden_dim=self.hidden_dim, edge_dim=self.num_rbf)
            for _ in range(self.num_hidden_layers)
        ])

        self.mp_layers = nn.ModuleList([
            ComplexMessagePassing(
                hidden_dim=self.hidden_dim,
                edge_dim=self.num_rbf,
                use_angular=self.use_angular,
                num_angle_basis=self.num_angle_basis,
                angular_scale_init=self.angular_scale_init,
            )
            for _ in range(self.num_hidden_layers)
        ])

        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.hidden_dim) for _ in range(self.num_hidden_layers)
            ])
        else:
            self.layer_norms = None

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # pool_dim = 2H: cat([real_mean, imag_mean]) del pooling cartesiano
        pool_dim = self.hidden_dim * 2
        self.out_head = nn.Sequential(
            nn.Linear(pool_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            ShiftedSoftplus(),
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
        coords_cart = self._as_list_or_default(batch.get("coords_cart"), len(atom_feats))

        edge_index_list = self._as_list_or_default(batch.get("edge_index"), len(atom_feats))
        edge_attr_list = self._as_list_or_default(batch.get("edge_attr"), len(atom_feats))

        mol_outputs = []

        for feats, sph, cart, ei, ea in zip(atom_feats, coords_sph, coords_cart, edge_index_list, edge_attr_list):
            x = torch.cat([feats.float(), sph.float()], dim=-1)
            z = self.input_activation(self.embedding(x))

            if self._has_edges(ei, ea):
                rbf = self.rbf(ea.float())
            else:
                ei = None
                rbf = None

            for layer_idx in range(self.num_hidden_layers):
                z_new = self.attn_layers[layer_idx](z, edge_index=ei, rbf=rbf)

                if ei is not None and rbf is not None:
                    z_new = self.mp_layers[layer_idx](z_new, ei, rbf, coords_cart=cart)

                if self.use_residuals:
                    # Suma compleja correcta en espacio cartesiano.
                    # z_new + z ≠ (r_new+r, θ_new+θ): la suma de magnitudes y fases
                    # no corresponde a adición de números complejos.
                    r, theta = z.magnitude, z.phase
                    r_new, theta_new = z_new.magnitude, z_new.phase
                    res_real = r_new * torch.cos(theta_new) + r * torch.cos(theta)
                    res_imag = r_new * torch.sin(theta_new) + r * torch.sin(theta)
                    z_new.magnitude = torch.sqrt(res_real ** 2 + res_imag ** 2 + 1e-9)
                    z_new.phase = torch.atan2(res_imag, res_real)

                if self.use_layernorm and self.layer_norms is not None:
                    # LayerNorm estabiliza la escala; abs garantiza no-negatividad.
                    z_new.magnitude = torch.abs(
                        self.layer_norms[layer_idx](z_new.magnitude)
                    ).clamp_min(1e-9)

                z_new = self.complex_activations[layer_idx](z_new)
                z_new.magnitude = self.dropout(z_new.magnitude)
                z = z_new

            # Pooling en espacio cartesiano — preserva información de fase hasta el head.
            # mean(real) + mean(imag) mantiene pool_dim = 2H igual que antes.
            z_real = z.magnitude * torch.cos(z.phase)    # [N, H]
            z_imag = z.magnitude * torch.sin(z.phase)    # [N, H]
            mol_repr = torch.cat([z_real.mean(dim=0), z_imag.mean(dim=0)], dim=-1)  # [2H]
            mol_outputs.append(mol_repr)

        return self.out_head(torch.stack(mol_outputs))
