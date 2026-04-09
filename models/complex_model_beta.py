from .complex_layers import ComplexEmbedding, ComplexPolarAttention, RealProjection
import torch
import torch.nn as nn
import math


class ComplexPolarTransformerBeta(nn.Module):
    """
    Transformer complejo-polar para predicción de propiedades moleculares.

    Mejoras sobre la versión anterior:

    CAMBIO 1 — Scaling sqrt(D) en atención:
        El score de atención ahora se divide por sqrt(hidden_dim) antes
        del softmax. Sin esto, los scores crecen con la dimensión y
        saturan la función softmax, colapsando los gradientes hacia cero.
        Referencia: Vaswani et al. (2017), Attention Is All You Need.

    CAMBIO 2 — Pooling mean + sum:
        La versión anterior usaba solo sum(). Esto hace que la predicción
        dependa del número de átomos: una molécula con 10 átomos siempre
        predice valores mayores que una con 5. El pooling mean es invariante
        al tamaño, pero pierde información de extensividad (energías totales
        sí escalan con el número de átomos). La combinación mean+sum
        captura ambas propiedades simultáneamente.
        Referencia: Gilmer et al. (2017) MPNN for Quantum Chemistry.

    CAMBIO 3 — Uso de edge_attr como bias de atención:
        Las aristas del grafo molecular contienen distancias interatómicas.
        Ignorarlas descarta información geométrica crucial: la energía de
        interacción entre dos átomos depende fuertemente de su distancia.
        Ahora se proyectan a un escalar de bias que se suma al score de
        atención antes del softmax.
        Referencia: ViSNet (Wang et al., 2023).

    CAMBIO 4 — RealProjection usa real + imag:
        La proyección final ahora concatena parte real e imaginaria del
        tensor complejo en lugar de descartar la imaginaria. Esto preserva
        la información de fase en la predicción final.

    CAMBIO 5 — 4 capas de atención (antes 2):
        Más capas permiten que la información se propague entre átomos
        que no son vecinos directos, capturando interacciones de largo
        alcance. SchNet usa 6 capas de interacción. Referencia: [14].

    CAMBIO 6 — LayerNorm activado entre capas:
        Con 4 capas, las magnitudes complejas pueden crecer o colapsar
        sin normalización. LayerNorm estabiliza el entrenamiento profundo.
        Referencia: Ba et al. (2016) Layer Normalization.

    CAMBIO 7 — Cutoff suave por distancia en EdgeBias:
        EdgeBiasProjection ahora aprende una penalización proporcional
        a la distancia interatómica, con parámetro dist_scale inicializado
        negativo. Átomos lejanos reciben bias negativo → menos atención.
        Equivalente funcional al envelope de SchNet. Referencia: [14].

    Args:
        in_dim:           dimensión de atom_types (ej: 5)
        hidden_dim:       dimensión del espacio latente complejo
        out_dim:          número de propiedades a predecir (ej: 1)
        num_hidden_layers: capas de atención apiladas (default 4)
        edge_dim:         dimensión de edge_attr (default 4)
        dropout:          dropout entre capas (default 0.1)
        use_residuals:    conexiones residuales entre capas (default True)
        use_layernorm:    LayerNorm sobre magnitud entre capas (default True)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int = 1,
        num_hidden_layers: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.1,
        use_residuals: bool = True,
        use_layernorm: bool = True,
        **kwargs,   # compatibilidad con YAML
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_residuals = use_residuals
        self.use_layernorm = use_layernorm
        self.num_hidden_layers = num_hidden_layers

        # Entrada total = atom_types + (r, θ, φ)
        self.input_dim = in_dim + 3

        # Embedding complejo inicial
        self.embedding = ComplexEmbedding(
            in_dim=self.input_dim,
            hidden_dim=hidden_dim,
        )

        # Capas de atención apiladas
        # Cada capa recibe edge_dim para el bias de aristas (CAMBIO 3)
        self.attn_layers = nn.ModuleList([
            ComplexPolarAttention(hidden_dim=hidden_dim, edge_dim=edge_dim)
            for _ in range(num_hidden_layers)
        ])

        # LayerNorm sobre magnitud (opcional, aplicado entre capas)
        if use_layernorm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_hidden_layers)
            ])
        else:
            self.layer_norms = None

        # Dropout entre capas
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # CAMBIO 2 — pooling mean+sum: la capa de salida recibe
        # concatenación de mean y sum → input_dim = hidden_dim * 2 * 2
        # (hidden_dim*2 porque RealProjection ya concatena real+imag,
        #  y luego mean+sum duplica eso)
        pool_dim = hidden_dim * 2       # mean + sum concatenados (cada uno = hidden_dim)

        self.out_head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Proyección complejo → real por átomo
        # (hidden_dim * 2 por concatenar real+imag — CAMBIO 4)
        self.out_proj = RealProjection(hidden_dim, hidden_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict con claves
                - atom_types:       List[Tensor (N_i, in_dim)]
                - coords_spherical: List[Tensor (N_i, 3)]
                - edge_index:       List[Tensor (2, E_i)]
                - edge_attr:        List[Tensor (E_i, edge_dim)]
        Returns:
            Tensor [B, out_dim]
        """
        atom_feats  = batch["atom_types"]
        coords_sph  = batch["coords_spherical"]
        edge_index  = batch.get("edge_index", [None] * len(atom_feats))
        edge_attr   = batch.get("edge_attr",  [None] * len(atom_feats))

        mol_outputs = []

        for feats, sph, ei, ea in zip(atom_feats, coords_sph, edge_index, edge_attr):
            # feats: [N, in_dim]  sph: [N, 3]  ei: [2, E]  ea: [E, edge_dim]

            # Si edge_attr tiene dim distinta a la esperada, dar error claro
            if ea is not None and ea.dim() == 2:
                actual_dim = ea.shape[-1]
                expected_dim = self.attn_layers[0].edge_bias.proj[0].in_features
                if actual_dim != expected_dim:
                    raise ValueError(
                        f"edge_attr tiene {actual_dim} features por arista pero el modelo "
                        f"fue construido con edge_dim={expected_dim}. "
                        f"Ajusta edge_dim: {actual_dim} en el YAML."
                    )

            # Concatenar features químicas y coordenadas polares
            x = torch.cat([feats.float(), sph.float()], dim=-1)  # [N, in_dim+3]

            # Embedding complejo inicial
            z = self.embedding(x)   # ComplexTensor [N, D]

            # Capas de atención apiladas con residual opcional
            for layer_idx, attn in enumerate(self.attn_layers):

                z_new = attn(z, edge_index=ei, edge_attr=ea)

                # CAMBIO 1 implícito: el scaling sqrt(D) está dentro de attn

                # Conexión residual (CAMBIO arquitectural del YAML)
                if self.use_residuals:
                    z_new.magnitude = z_new.magnitude + z.magnitude
                    z_new.phase     = z_new.phase     + z.phase

                # LayerNorm opcional sobre magnitud
                if self.use_layernorm and self.layer_norms is not None:
                    z_new.magnitude = self.layer_norms[layer_idx](z_new.magnitude)

                # Dropout sobre magnitud
                z_new.magnitude = self.dropout(z_new.magnitude)

                z = z_new

            # Proyección por átomo: ComplexTensor → real [N, hidden_dim]
            atom_repr = self.out_proj(z)   # [N, hidden_dim]

            # CAMBIO 2 — Pooling mean + sum concatenados
            pool_mean = atom_repr.mean(dim=0)   # [hidden_dim]
            pool_sum  = atom_repr.sum(dim=0)    # [hidden_dim]
            mol_repr  = torch.cat([pool_mean, pool_sum], dim=-1)  # [hidden_dim*2]

            mol_outputs.append(mol_repr)

        # Stack y predicción final
        mol_batch = torch.stack(mol_outputs)          # [B, hidden_dim*2]
        out = self.out_head(mol_batch)                # [B, out_dim]
        return out