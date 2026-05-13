import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_tensor import ComplexTensor


# ─────────────────────────────────────────────
# modReLU — activación nativa para dominio polar
# ─────────────────────────────────────────────
class ModReLU(nn.Module):
    """
    Activación polar: modifica SOLO la magnitud, preserva la fase.

        modReLU(r, θ) = (ReLU(r + bias), θ)

    Bias learnable por feature. Preserva la información de orientación
    completa, alineado con el anteproyecto (representación polar).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.clamp(magnitude + self.bias, min=0.0)


# ─────────────────────────────────────────────
# RBFExpansion con features angulares polares
# ─────────────────────────────────────────────
class RBFExpansion(nn.Module):
    """
    Expansión radial + angular para aristas moleculares.

    Usa edge_attr con 4 columnas:
        [:, 0]  distancia real d_ij en Å         → RBF gaussianas + cosine cutoff
        [:, 1]  Δr normalizado                   → ignorado (redundante con dist)
        [:, 2]  sin(Δθ)  diferencia polar         → codificación sin+cos
        [:, 3]  sin(Δφ)  diferencia azimutal      → codificación sin+cos

    Output: [E, num_rbf + 4]  (RBF radiales + 4 features angulares)
    """

    def __init__(self, num_rbf: int = 150, cutoff: float = 7.0):
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf debe ser >= 1")
        if cutoff <= 0:
            raise ValueError("cutoff debe ser > 0")

        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)

        centers = torch.linspace(0.0, self.cutoff, self.num_rbf)
        self.register_buffer("centers", centers)

        spacing = self.cutoff / (self.num_rbf - 1) if self.num_rbf > 1 else self.cutoff
        self.gamma = 1.0 / (spacing ** 2)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr no puede ser None en RBFExpansion.forward")
        if edge_attr.numel() == 0:
            return edge_attr.new_empty((0, self.num_rbf + 4))

        dist   = edge_attr[:, 0].float().unsqueeze(-1)   # [E, 1]  Å
        sin_dt = edge_attr[:, 2].float().unsqueeze(-1)   # [E, 1]  sin(Δθ)
        sin_dp = edge_attr[:, 3].float().unsqueeze(-1)   # [E, 1]  sin(Δφ)

        # RBF radiales con cosine cutoff
        rbf = torch.exp(-self.gamma * (dist - self.centers) ** 2)
        inside = (dist < self.cutoff).float()
        cos_cut = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * inside
        rbf = rbf * cos_cut                              # [E, num_rbf]

        # Codificación angular sin+cos (evita discontinuidades en ±π)
        cos_dt = torch.sqrt((1.0 - sin_dt.clamp(-1.0, 1.0) ** 2).clamp(min=0.0))
        cos_dp = torch.sqrt((1.0 - sin_dp.clamp(-1.0, 1.0) ** 2).clamp(min=0.0))
        ang = torch.cat([sin_dt, cos_dt, sin_dp, cos_dp], dim=-1)  # [E, 4]

        return torch.cat([rbf, ang], dim=-1)             # [E, num_rbf + 4]


# ─────────────────────────────────────────────
# ComplexEmbedding
# ─────────────────────────────────────────────
class ComplexEmbedding(nn.Module):
    """Convierte features reales en representación compleja polar."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.to_mag = nn.Linear(in_dim, hidden_dim)
        self.to_phase = nn.Linear(in_dim, hidden_dim)
        self.modrelu = ModReLU(hidden_dim)

    def forward(self, x: torch.Tensor) -> ComplexTensor:
        magnitude = self.modrelu(F.softplus(self.to_mag(x)))
        phase = torch.tanh(self.to_phase(x)) * math.pi
        return ComplexTensor(magnitude, phase)


# ─────────────────────────────────────────────
# ComplexMessagePassing con modReLU
# ─────────────────────────────────────────────
class ComplexMessagePassing(nn.Module):
    """
    Paso de mensajes complejo ponderado por RBF + features angulares.

    Usa modReLU como activación sobre la magnitud del mensaje,
    preservando la fase (orientación) sin distorsión.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        # Proyección de edge features → espacio oculto
        self.edge_to_hidden = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # modReLU para la magnitud del mensaje (preserva fase)
        self.msg_modrelu = ModReLU(hidden_dim)

        # Proyección de fase del mensaje
        self.edge_to_phase = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.Tanh(),
        )
        self.phase_scale = nn.Parameter(torch.tensor(math.pi))

        # Compuerta de actualización (opera sobre reales — Sigmoid correcto)
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # Normalización independiente de magnitud y fase
        self.norm_mag = nn.LayerNorm(hidden_dim)
        self.norm_phase = nn.LayerNorm(hidden_dim)

    def forward(self, cpx: ComplexTensor, edge_index: torch.Tensor, rbf: torch.Tensor) -> ComplexTensor:
        if edge_index is None or rbf is None or edge_index.numel() == 0 or rbf.numel() == 0:
            return cpx

        mag = cpx.magnitude
        phase = cpx.phase
        src, dst = edge_index[0], edge_index[1]

        # Mensaje de magnitud: edge features proyectadas × magnitud vecino → modReLU
        h_edge = self.edge_to_hidden(rbf)                    # [E, D]
        msg_mag = self.msg_modrelu(h_edge * mag[src])        # modReLU polar [E, D]

        # Mensaje de fase: desplazamiento angular aprendido
        msg_phase = self.edge_to_phase(rbf) * self.phase_scale + phase[src]  # [E, D]

        # Agregación (scatter_add)
        agg_mag = torch.zeros_like(mag)
        agg_phase = torch.zeros_like(phase)
        agg_mag.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_mag), msg_mag)
        agg_phase.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_phase), msg_phase)

        # Compuerta + normalización de magnitud
        gate = self.update_gate(torch.cat([mag, agg_mag], dim=-1))
        new_mag = self.norm_mag(mag + gate * agg_mag)

        # Normalización de fase + reacotado a [-π, π]
        new_phase = torch.tanh(self.norm_phase(phase + gate * agg_phase)) * math.pi

        return ComplexTensor(new_mag, new_phase)


# ─────────────────────────────────────────────
# EdgeBiasProjection
# ─────────────────────────────────────────────
class EdgeBiasProjection(nn.Module):
    """Proyecta RBF+angular a un escalar de bias para atención."""

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, rbf: torch.Tensor) -> torch.Tensor:
        return self.proj(rbf).squeeze(-1)


# ─────────────────────────────────────────────
# ComplexPolarAttention — O(E) sparse (no O(N²))
# ─────────────────────────────────────────────
class ComplexPolarAttention(nn.Module):
    """
    Atención compleja-polar sparse sobre edges definidos.

    Calcula scores SOLO sobre las aristas del grafo molecular (O(E)),
    no sobre todos los pares N×N (O(N²)). Usa scatter softmax por
    nodo destino para normalizar los pesos de atención.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 154):
        super().__init__()
        self.scale = math.sqrt(hidden_dim)
        self.edge_bias = EdgeBiasProjection(edge_dim, hidden_dim)

    def forward(
        self,
        cpx: ComplexTensor,
        edge_index: torch.Tensor = None,
        rbf: torch.Tensor = None,
    ) -> ComplexTensor:
        mag = cpx.magnitude
        phase = cpx.phase

        if edge_index is None or edge_index.numel() == 0:
            return cpx

        src, dst = edge_index[0], edge_index[1]

        # Score polar: producto interno complejo sobre edges — O(E)
        phase_diff = phase[src] - phase[dst]                         # [E, D]
        mag_prod   = mag[src]   * mag[dst]                           # [E, D]
        scores = (mag_prod * torch.cos(phase_diff)).sum(-1) / self.scale  # [E]

        # Bias aprendido desde RBF + features angulares
        if rbf is not None and rbf.numel() > 0:
            scores = scores + self.edge_bias(rbf)

        # Scatter softmax por nodo destino (estable)
        scores_shifted = scores - scores.max()
        exp_s = torch.exp(scores_shifted)
        denom = torch.zeros(mag.shape[0], device=mag.device, dtype=mag.dtype)
        denom.scatter_add_(0, dst, exp_s)
        attn = exp_s / (denom[dst] + 1e-9)                          # [E]

        # Agregación ponderada de magnitud y fase
        new_mag   = torch.zeros_like(mag)
        new_phase = torch.zeros_like(phase)
        new_mag.scatter_add_(
            0, dst.unsqueeze(1).expand_as(mag[src]),
            attn.unsqueeze(1) * mag[src]
        )
        new_phase.scatter_add_(
            0, dst.unsqueeze(1).expand_as(phase[src]),
            attn.unsqueeze(1) * phase[src]
        )

        return ComplexTensor(new_mag, new_phase)


# ─────────────────────────────────────────────
# RealProjection
# ─────────────────────────────────────────────
class RealProjection(nn.Module):
    """Proyecta ComplexTensor a reales concatenando parte real + imaginaria."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim * 2, out_dim)

    def forward(self, cpx: ComplexTensor) -> torch.Tensor:
        z = cpx.as_cartesian()
        return self.lin(torch.cat([z.real, z.imag], dim=-1))
