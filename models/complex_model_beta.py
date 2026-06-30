import torch
import torch.nn as nn

from .complex_layers import (
    RBFExpansion,
    ComplexEmbedding,
    ComplexPolarAttention,
    ComplexMessagePassing,
    MagnitudeRMSNorm,
    ShiftedSoftplus,
    ModReLU,
)


class ComplexPolarTransformerBeta(nn.Module):
    """
    ComplexPolarTransformer — versión benchmark v12.

    Cambios sobre v11 (auditoría científica):
    - Invarianza rotacional restaurada: se elimina θ y φ del embedding inicial.
      Solo se usa r (distancia al centroide, invariante a rotación). La información
      angular entra vía AngularBasis en ComplexMessagePassing, que usa ángulos
      entre enlaces (cos θ_jik), los cuales SÍ son invariantes a SE(3).
    - Per-atom readout: MLP aplicado por átomo → media. Corrige cancelación de
      fases opuestas en el pooling global.
    - Atomic energy offsets aprendibles por tipo de átomo (H,C,N,O,F).
    - _phase_reg: concentración circular promedio de la última capa para detección
      y regularización de colapso de fase.
    """

    def __init__(
        self,
        in_dim:             int   = 12,    # 5 one-hot + 3 hibrid + arom + ring + charge + Hs
        hidden_dim:         int   = 256,
        out_dim:            int   = 1,
        num_hidden_layers:  int   = 6,
        num_heads:          int   = 4,
        num_rbf:            int   = 64,
        cutoff:             float = 5.0,
        edge_dim:           int   = 4,
        dropout:            float = 0.1,
        use_residuals:      bool  = True,
        use_layernorm:      bool  = True,
        activation:         str   = "modrelu",
        modrelu_init_bias:  float = 5.0,
        modrelu_eps:        float = 1e-8,
        use_angular:        bool  = False,
        num_angle_basis:    int   = 16,
        angular_scale_init: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim        = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_heads         = int(num_heads)
        self.num_rbf           = int(num_rbf)
        self.cutoff            = float(cutoff)
        self.use_residuals     = bool(use_residuals)
        self.use_layernorm     = bool(use_layernorm)
        self.activation_name   = str(activation).lower().strip()
        self.use_angular       = bool(use_angular)
        self.num_angle_basis   = int(num_angle_basis)
        self.angular_scale_init = float(angular_scale_init)

        # input_dim: atom features (in_dim) + r solamente.
        # Se elimina θ y φ del centroide porque no son invariantes a rotación:
        # dos moléculas idénticas pero rotadas tendrían valores distintos de θ,φ
        # lo que rompe la generalización. La información angular correcta entra
        # solo a través de AngularBasis (ángulos entre enlaces, sí invariantes).
        self.input_dim = int(in_dim) + 1   # +1 = distancia al centroide r

        # ── Capas radiales y de embedding ────────────────────────────────
        self.rbf       = RBFExpansion(num_rbf=self.num_rbf, cutoff=self.cutoff)
        self.embedding = ComplexEmbedding(self.input_dim, self.hidden_dim)

        # ── Activaciones ─────────────────────────────────────────────────
        if self.activation_name == "modrelu":
            self.input_activation = ModReLU(
                self.hidden_dim,
                init_bias=float(modrelu_init_bias),
                eps=float(modrelu_eps),
            )
            self.complex_activations = nn.ModuleList([
                ModReLU(self.hidden_dim, init_bias=float(modrelu_init_bias), eps=float(modrelu_eps))
                for _ in range(self.num_hidden_layers)
            ])
        elif self.activation_name in {"identity", "none", "linear"}:
            self.input_activation    = nn.Identity()
            self.complex_activations = nn.ModuleList(
                [nn.Identity() for _ in range(self.num_hidden_layers)]
            )
        else:
            raise ValueError(
                f"activation no soportada: '{activation}'. Usa 'modrelu' o 'identity'."
            )

        # ── Capas de atención: multi-head Q/K/V con edge-masking ─────────
        self.attn_layers = nn.ModuleList([
            ComplexPolarAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                edge_dim=self.num_rbf,
            )
            for _ in range(self.num_hidden_layers)
        ])

        # ── Capas de paso de mensajes ─────────────────────────────────────
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

        # ── Normalización de magnitudes (MagnitudeRMSNorm) ────────────────
        if self.use_layernorm:
            self.layer_norms = nn.ModuleList([
                MagnitudeRMSNorm(self.hidden_dim)
                for _ in range(self.num_hidden_layers)
            ])
        else:
            self.layer_norms = None

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # ── Readout MLP (per-atom) ────────────────────────────────────────
        # Aplicado a cada átomo individualmente: out = mean_i(MLP(z_i)).
        # Evita cancelación de fases: mean([r·cos0, r·cosπ]) = 0 destruye info.
        # Con per-atom MLP, cada átomo contribuye su energía desde su
        # representación completa, luego se promedia (extensividad correcta).
        pool_dim = self.hidden_dim * 2   # cat(real, imag) por átomo [N, 2H]
        self.out_head = nn.Sequential(
            nn.Linear(pool_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim // 2, out_dim),
        )

        # ── Offsets de energía atómica aprendibles ───────────────────────
        # Análogo al atomic energy baseline de SchNet/PaiNN.
        # ε_type aprende la contribución energética media por elemento.
        # La GNN solo necesita aprender la corrección estructural residual.
        # Inicializados a 0: sin perturbación al inicio del entrenamiento.
        self._n_atom_types = 5  # H, C, N, O, F (QM9)
        self.atom_energy_offset = nn.Embedding(self._n_atom_types + 1, out_dim)
        nn.init.zeros_(self.atom_energy_offset.weight)

        # Buffer para regularización de fase (calculado en forward)
        self._phase_reg: torch.Tensor | None = None

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
        atom_feats      = batch["atom_types"]
        coords_sph      = batch["coords_spherical"]
        coords_cart     = self._as_list_or_default(batch.get("coords_cart"),  len(atom_feats))
        edge_index_list = self._as_list_or_default(batch.get("edge_index"),   len(atom_feats))
        edge_attr_list  = self._as_list_or_default(batch.get("edge_attr"),    len(atom_feats))

        mol_outputs  = []
        phase_concs  = []   # concentración circular por molécula (para _phase_reg)

        for feats, sph, cart, ei, ea in zip(
            atom_feats, coords_sph, coords_cart, edge_index_list, edge_attr_list
        ):
            # Solo r (distancia al centroide): invariante a rotación.
            # θ y φ se descartan; la orientación entra solo vía AngularBasis.
            r = sph[:, 0:1].float()                                # [N, 1]
            x = torch.cat([feats.float(), r], dim=-1)              # [N, in_dim+1]
            z = self.input_activation(self.embedding(x))

            if self._has_edges(ei, ea):
                rbf = self.rbf(ea.float())                         # [E, num_rbf]
            else:
                ei  = None
                rbf = None

            for layer_idx in range(self.num_hidden_layers):
                # Atención multi-head con Q/K/V y edge-masking geométrico
                z_new = self.attn_layers[layer_idx](z, edge_index=ei, rbf=rbf)

                # Paso de mensajes (radial + angular si use_angular=True)
                if ei is not None and rbf is not None:
                    z_new = self.mp_layers[layer_idx](z_new, ei, rbf, coords_cart=cart)

                # Residual complejo en espacio cartesiano
                if self.use_residuals:
                    r_m,   theta     = z.magnitude,     z.phase
                    r_new, theta_new = z_new.magnitude, z_new.phase
                    res_real = r_new * torch.cos(theta_new) + r_m * torch.cos(theta)
                    res_imag = r_new * torch.sin(theta_new) + r_m * torch.sin(theta)
                    z_new.magnitude = torch.sqrt(res_real ** 2 + res_imag ** 2 + 1e-9)
                    z_new.phase     = torch.atan2(res_imag, res_real)

                # MagnitudeRMSNorm — preserva no-negatividad sin abs()
                if self.use_layernorm and self.layer_norms is not None:
                    z_new.magnitude = self.layer_norms[layer_idx](z_new.magnitude).clamp_min(1e-9)

                z_new = self.complex_activations[layer_idx](z_new)
                z_new.magnitude = self.dropout(z_new.magnitude)
                z = z_new

            # ── Concentración circular de fase (última capa) ──────────────
            # R = |mean(e^{iθ})| ∈ [0,1]:  0 = fases diversas, 1 = colapso total.
            # Se calcula por dimensión y se promedia → escalar diferenciable.
            cos_m = z.phase.cos().mean(dim=0)                      # [H]
            sin_m = z.phase.sin().mean(dim=0)                      # [H]
            concentration = torch.sqrt(cos_m ** 2 + sin_m ** 2 + 1e-8).mean()
            phase_concs.append(concentration)

            # ── Per-atom readout ──────────────────────────────────────────
            z_real = z.magnitude * torch.cos(z.phase)              # [N, H]
            z_imag = z.magnitude * torch.sin(z.phase)              # [N, H]

            # Offset atómico por tipo de elemento
            one_hot5 = feats[:, :self._n_atom_types].float()       # [N, 5]
            has_type = one_hot5.sum(dim=-1) > 0.5
            type_idx = one_hot5.argmax(dim=-1)
            type_idx = type_idx.masked_fill(~has_type, self._n_atom_types)
            offsets  = self.atom_energy_offset(type_idx)           # [N, out_dim]

            atom_input    = torch.cat([z_real, z_imag], dim=-1)    # [N, 2H]
            atom_energies = self.out_head(atom_input) + offsets     # [N, out_dim]
            mol_outputs.append(atom_energies.mean(dim=0))           # [out_dim]

        # Almacena concentración media del batch para que el trainer la use
        # como pérdida auxiliar de regularización (sin detach → gradiente activo)
        self._phase_reg = torch.stack(phase_concs).mean() if phase_concs else None

        return torch.stack(mol_outputs)                             # [B, out_dim]
