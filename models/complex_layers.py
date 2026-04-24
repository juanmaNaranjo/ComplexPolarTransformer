import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .complex_tensor import ComplexTensor


# ============================================================
# RBF — Radial Basis Function Expansion  (NUEVO en v6)
#
# Convierte la distancia escalar d_ij en un vector de num_rbf
# valores gaussianos centrados uniformemente en [0, cutoff].
#
# e_k(d) = exp(-γ (d - μ_k)²)   con  γ = 1/spacing²
#
# Esto es el componente más crítico de la v6. En SchNet [14],
# esta expansión permite al modelo aprender cualquier función
# radial suave, incluyendo el potencial de Lennard-Jones, la
# atracción de van der Waals, y los perfiles de enlace covalente.
# Sin RBF, el modelo solo ve 4 números crudos de edge_attr y
# no puede distinguir diferencias de distancia menores a ~0.1 Å.
#
# Aquí se incluye además el cosine cutoff de SchNet:
#
#   f_cut(d) = 0.5 × (cos(π × d/cutoff) + 1)   si d < cutoff
#            = 0                                  si d ≥ cutoff
#
# que suaviza la contribución de pares lejanos hasta cero en
# la frontera del cutoff, evitando discontinuidades.
#
# Referencia: Schütt et al. (2018) SchNet [14].
# ============================================================

class RBFExpansion(nn.Module):
    """
    Expansión de distancia interatómica en base de gaussianas.

    Transforma edge_attr[:, 0] (distancia) en un vector de
    num_rbf features continuas y físicamente significativas.

    Args:
        num_rbf:  número de gaussianas (default 50)
        cutoff:   distancia máxima en Å (default 5.0)
    """

    def __init__(self, num_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff  = cutoff

        # Centros uniformes en [0, cutoff] — fijos, no aprendibles
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer('centers', centers)

        # Ancho de cada gaussiana
        spacing = cutoff / (num_rbf - 1)
        self.gamma = 1.0 / (spacing ** 2)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: [E, edge_dim_original] — se usa solo edge_attr[:,0]
        Returns:
            rbf: [E, num_rbf] — representación radial expandida
        """
        dist = edge_attr[:, 0].float().unsqueeze(-1)   # [E, 1]

        # Expansión gaussiana
        rbf = torch.exp(-self.gamma * (dist - self.centers) ** 2)   # [E, num_rbf]

        # Cosine cutoff — suaviza a cero en la frontera
        cos_cutoff = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)
        cos_cutoff = cos_cutoff.clamp(0.0, 1.0)   # fuera del cutoff = 0

        return rbf * cos_cutoff   # [E, num_rbf]


# ============================================================
# Embedding complejo polar
# ============================================================

class ComplexEmbedding(nn.Module):
    """
    Convierte features reales en representación compleja polar.
    Magnitud >= 0 (softplus), fase en [-pi, pi] (tanh * pi).
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.to_mag   = nn.Linear(in_dim, hidden_dim)
        self.to_phase = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> ComplexTensor:
        magnitude = F.softplus(self.to_mag(x))
        phase     = torch.tanh(self.to_phase(x)) * math.pi
        return ComplexTensor(magnitude, phase)


# ============================================================
# Message Passing Complejo con RBF  (mejorado en v6)
#
# Igual que v3 pero ahora edge_dim = num_rbf (50) en vez de 4.
# Las RBF dan al MP una representación radial continua y rica,
# permitiendo que el modelo aprenda perfiles de interacción
# físicamente realistas.
# ============================================================

class ComplexMessagePassing(nn.Module):
    """
    Paso de mensajes complejo ponderado por RBF.

    msg_ij = φ_rbf(rbf_ij) ⊙ z_j    (multiplicación compleja)
    z_i   ← z_i + gate × Σ_j msg_ij

    Args:
        hidden_dim: dimensión D del espacio latente
        edge_dim:   dimensión de la representación de arista (= num_rbf)
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()

        self.edge_to_mag = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.edge_to_phase = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.phase_scale = nn.Parameter(torch.tensor(math.pi))
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm_mag = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        cpx:        ComplexTensor,
        edge_index: torch.Tensor,
        rbf:        torch.Tensor,   # [E, num_rbf] — ya expandido
    ) -> ComplexTensor:

        mag   = cpx.magnitude
        phase = cpx.phase
        src, dst = edge_index[0], edge_index[1]

        msg_mag   = self.edge_to_mag(rbf)   * mag[src]
        msg_phase = self.edge_to_phase(rbf) * self.phase_scale + phase[src]

        agg_mag   = torch.zeros_like(mag)
        agg_phase = torch.zeros_like(phase)
        agg_mag.scatter_add_(  0, dst.unsqueeze(1).expand_as(msg_mag),   msg_mag)
        agg_phase.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_phase), msg_phase)

        gate      = self.update_gate(torch.cat([mag, agg_mag], dim=-1))
        new_mag   = self.norm_mag(mag   + gate * agg_mag)
        new_phase = phase + gate * agg_phase

        return ComplexTensor(new_mag, new_phase)


# ============================================================
# EdgeBias para atención — usa RBF en vez de edge_attr raw
# ============================================================

class EdgeBiasProjection(nn.Module):
    """
    Proyecta RBF a un escalar de bias para la atención.
    El cosine cutoff ya está incorporado en el RBF — no necesita
    penalización adicional por distancia.
    """

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, rbf: torch.Tensor) -> torch.Tensor:
        return self.proj(rbf).squeeze(-1)


# ============================================================
# Atención compleja-polar con RBF
# ============================================================

class ComplexPolarAttention(nn.Module):
    """
    Atención compleja con bias de aristas basado en RBF.
    score_ij = Σ |z_i||z_j| cos(θ_i-θ_j) / sqrt(D) + rbf_bias_ij
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 50):
        super().__init__()
        self.scale     = math.sqrt(hidden_dim)
        self.edge_bias = EdgeBiasProjection(edge_dim, hidden_dim)

    def forward(
        self,
        cpx:        ComplexTensor,
        edge_index: torch.Tensor = None,
        rbf:        torch.Tensor = None,
    ) -> ComplexTensor:

        mag   = cpx.magnitude
        phase = cpx.phase

        phase_diff = phase.unsqueeze(1) - phase.unsqueeze(0)
        mag_prod   = mag.unsqueeze(1)   * mag.unsqueeze(0)
        scores     = torch.sum(mag_prod * torch.cos(phase_diff), dim=-1) / self.scale

        if edge_index is not None and rbf is not None:
            bias_vals = self.edge_bias(rbf)
            i_idx, j_idx = edge_index[0], edge_index[1]
            scores = scores.clone()
            scores[i_idx, j_idx] += bias_vals

        attn_weights = F.softmax(scores, dim=1)
        new_mag      = torch.matmul(attn_weights, mag)
        new_phase    = torch.matmul(attn_weights, phase)

        return ComplexTensor(new_mag, new_phase)


# ============================================================
# Proyección complejo → real
# ============================================================

class RealProjection(nn.Module):
    """Proyecta ComplexTensor a reales concatenando real + imag."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim * 2, out_dim)

    def forward(self, cpx: ComplexTensor) -> torch.Tensor:
        z = cpx.as_cartesian()
        return self.lin(torch.cat([z.real, z.imag], dim=-1))