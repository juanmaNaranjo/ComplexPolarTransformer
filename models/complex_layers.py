import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .complex_tensor import ComplexTensor


# ============================================================
# Embedding complejo polar
# ============================================================

class ComplexEmbedding(nn.Module):
    """
    Convierte features reales en representación compleja polar.
    """

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.to_mag = nn.Linear(in_dim, hidden_dim)
        self.to_phase = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        # Magnitud positiva
        magnitude = F.softplus(self.to_mag(x))
        # Fase acotada a [-pi, pi]
        phase = torch.tanh(self.to_phase(x)) * math.pi
        return ComplexTensor(magnitude, phase)


# ============================================================
# Atención compleja-polar (versión mínima defendible)
# ============================================================

class ComplexPolarAttention(nn.Module):
    """
    Atención compleja basada en magnitud y fase.

    score_ij = |z_i||z_j| cos(θ_i - θ_j)
    """

    def __init__(self):
        super().__init__()

    def forward(self, cpx: ComplexTensor):
        """
        Args:
            cpx: ComplexTensor con shape [N, D]
        Returns:
            ComplexTensor actualizado
        """
        mag = cpx.magnitude        # [N, D]
        phase = cpx.phase          # [N, D]

        # Score escalar por par de átomos
        # [N, N]
        phase_diff = phase.unsqueeze(1) - phase.unsqueeze(0)
        mag_prod = mag.unsqueeze(1) * mag.unsqueeze(0)

        scores = torch.sum(mag_prod * torch.cos(phase_diff), dim=-1)

        # Normalización tipo atención
        attn_weights = F.softmax(scores, dim=1)  # [N, N]

        # Agregación compleja
        new_mag = torch.matmul(attn_weights, mag)
        new_phase = torch.matmul(attn_weights, phase)

        return ComplexTensor(new_mag, new_phase)


# ============================================================
# Proyección complejo → real
# ============================================================

class RealProjection(nn.Module):
    """
    Proyecta un tensor complejo a valores reales.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(dim, out_dim)

    def forward(self, cpx: ComplexTensor):
        z = cpx.as_cartesian()
        real = torch.real(z)
        return self.lin(real)
