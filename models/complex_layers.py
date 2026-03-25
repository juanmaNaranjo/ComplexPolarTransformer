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
    Atención compleja-polar con edge_attr geométrico.
    score_ij = |z_i||z_j| cos(θ_i - θ_j) + w · edge_feat_ij
    """

    def __init__(self, hidden_dim, edge_dim=4):
        super().__init__()
        # Proyecta edge_attr real al espacio de la atención
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

    def forward(self, cpx: ComplexTensor, edge_index=None, edge_attr=None):
        mag = cpx.magnitude    # [N, D]
        phase = cpx.phase      # [N, D]

        # Score base entre todos los pares
        phase_diff = phase.unsqueeze(1) - phase.unsqueeze(0)   # [N, N, D]
        mag_prod   = mag.unsqueeze(1) * mag.unsqueeze(0)       # [N, N, D]
        scores = torch.sum(mag_prod * torch.cos(phase_diff), dim=-1)  # [N, N]

        # Sumar contribución de edge_attr si está disponible
        if edge_index is not None and edge_attr is not None:
            edge_bias = torch.zeros(
                mag.size(0), mag.size(0),
                device=mag.device
            )
            proj = self.edge_proj(edge_attr)          # [E, D]
            edge_scores = proj.sum(dim=-1)            # [E]
            src, dst = edge_index[0], edge_index[1]
            edge_bias[src, dst] = edge_scores
            scores = scores + edge_bias

        attn_weights = F.softmax(scores, dim=1)       # [N, N]

        new_mag   = torch.matmul(attn_weights, mag)
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

class ComplexLayerNorm(nn.Module):
    """
    LayerNorm separado sobre magnitud y fase.
    Estabiliza el entrenamiento en el dominio complejo.
    """
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.norm_mag   = nn.LayerNorm(hidden_dim, eps=eps)
        self.norm_phase = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, cpx: ComplexTensor) -> ComplexTensor:
        mag_n   = self.norm_mag(cpx.magnitude)
        phase_n = self.norm_phase(cpx.phase)
        return ComplexTensor(F.softplus(mag_n), torch.tanh(phase_n) * math.pi)
    
class ModReLU(nn.Module):
    """
    modReLU(z) = ReLU(|z| + b) * exp(iθ)
    Activación compleja que preserva la fase.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, cpx: ComplexTensor) -> ComplexTensor:
        new_mag = F.relu(cpx.magnitude + self.bias)
        return ComplexTensor(new_mag, cpx.phase)


class ComplexFFN(nn.Module):
    """
    Feed-forward complejo: dos lineales sobre parte real e imaginaria
    con modReLU en el medio.
    """
    def __init__(self, hidden_dim, expansion=2):
        super().__init__()
        expanded = hidden_dim * expansion
        self.lin1_real = nn.Linear(hidden_dim, expanded)
        self.lin1_imag = nn.Linear(hidden_dim, expanded)
        self.act        = ModReLU(expanded)
        self.lin2_real  = nn.Linear(expanded, hidden_dim)
        self.lin2_imag  = nn.Linear(expanded, hidden_dim)

    def forward(self, cpx: ComplexTensor) -> ComplexTensor:
        real = cpx.magnitude * torch.cos(cpx.phase)
        imag = cpx.magnitude * torch.sin(cpx.phase)

        r1 = self.lin1_real(real) - self.lin1_imag(imag)
        i1 = self.lin1_real(imag) + self.lin1_imag(real)

        mid = ComplexTensor.from_cartesian(torch.complex(r1, i1))
        mid = self.act(mid)

        r2 = self.lin2_real(mid.magnitude * torch.cos(mid.phase)) \
           - self.lin2_imag(mid.magnitude * torch.sin(mid.phase))
        i2 = self.lin2_real(mid.magnitude * torch.sin(mid.phase)) \
           + self.lin2_imag(mid.magnitude * torch.cos(mid.phase))

        return ComplexTensor.from_cartesian(torch.complex(r2, i2))
