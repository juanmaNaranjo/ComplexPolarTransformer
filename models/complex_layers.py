import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_tensor import ComplexTensor


class ShiftedSoftplus(nn.Module):
    """
    Activación suave usada en modelos atomísticos tipo SchNet:
        shifted_softplus(x) = softplus(x) - log(2)
    Conserva suavidad para funciones radiales y centra la salida cerca de 0.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - math.log(2.0)




class ModReLU(nn.Module):
    """
    modReLU para tensores complejos en representacion polar.

    Definicion implementada:
        modReLU(z; b) = ReLU(|z| + b) * z / |z|, si |z| > eps
                       = 0, si |z| <= eps

    - Conserva la fase del numero complejo en la region activa.
    - Aprende un sesgo real por canal oculto.
    - Convierte internamente a cartesiano para garantizar radio positivo incluso
      si una normalizacion previa produjo magnitudes firmadas.
    """

    def __init__(self, hidden_dim: int, init_bias: float = -0.1, eps: float = 1e-8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.bias = nn.Parameter(torch.full((self.hidden_dim,), float(init_bias)))

    def forward(self, cpx: ComplexTensor) -> ComplexTensor:
        z = cpx.as_cartesian()
        radius = torch.abs(z)
        phase = torch.angle(z)

        # Convencion estable: para z=0 la salida es 0, evitando division por |z|.
        nonzero = (radius > self.eps).to(radius.dtype)
        activated_radius = F.relu(radius + self.bias) * nonzero

        # La fase solo importa cuando la magnitud es activa.
        active = (activated_radius > 0).to(phase.dtype)
        activated_phase = phase * active

        return ComplexTensor(activated_radius, activated_phase)


class RBFExpansion(nn.Module):
    """
    Expansión radial física para distancias interatómicas en Å.

    edge_attr[:, 0] debe ser la distancia real d_ij en Å.
    Devuelve num_rbf gaussianas centradas uniformemente en [0, cutoff] y moduladas
    por cosine cutoff:

        f_cut(d) = 0.5 * (cos(pi*d/cutoff) + 1), si d < cutoff
                 = 0, si d >= cutoff
    """

    def __init__(self, num_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf debe ser >= 1")
        if cutoff <= 0:
            raise ValueError("cutoff debe ser > 0")

        self.num_rbf = int(num_rbf)
        self.cutoff = float(cutoff)

        centers = torch.linspace(0.0, self.cutoff, self.num_rbf)
        self.register_buffer("centers", centers)

        if self.num_rbf == 1:
            spacing = self.cutoff
        else:
            spacing = self.cutoff / (self.num_rbf - 1)
        self.gamma = 1.0 / (spacing ** 2)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr no puede ser None en RBFExpansion.forward")
        if edge_attr.numel() == 0:
            return edge_attr.new_empty((0, self.num_rbf))

        dist = edge_attr[:, 0].float().unsqueeze(-1)  # [E, 1], distancia real en Å
        rbf = torch.exp(-self.gamma * (dist - self.centers) ** 2)

        inside = (dist < self.cutoff).float()
        cos_cutoff = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)
        cos_cutoff = cos_cutoff * inside

        return rbf * cos_cutoff


class ComplexEmbedding(nn.Module):
    """Convierte features reales en representación compleja polar."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.to_mag = nn.Linear(in_dim, hidden_dim)
        self.to_phase = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> ComplexTensor:
        magnitude = F.softplus(self.to_mag(x))
        phase = torch.tanh(self.to_phase(x)) * math.pi
        return ComplexTensor(magnitude, phase)


class ComplexMessagePassing(nn.Module):
    """
    Paso de mensajes complejo ponderado por RBF.

    msg_ij = phi_rbf(rbf_ij) * z_j
    z_i <- z_i + gate * sum_j msg_ij
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.edge_to_mag = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
        )
        self.edge_to_phase = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.phase_scale = nn.Parameter(torch.tensor(math.pi))
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm_mag = nn.LayerNorm(hidden_dim)

    def forward(self, cpx: ComplexTensor, edge_index: torch.Tensor, rbf: torch.Tensor) -> ComplexTensor:
        if edge_index is None or rbf is None or edge_index.numel() == 0 or rbf.numel() == 0:
            return cpx

        mag = cpx.magnitude
        phase = cpx.phase
        src, dst = edge_index[0], edge_index[1]

        msg_mag = self.edge_to_mag(rbf) * mag[src]
        msg_phase = self.edge_to_phase(rbf) * self.phase_scale + phase[src]

        agg_mag = torch.zeros_like(mag)
        agg_phase = torch.zeros_like(phase)
        agg_mag.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_mag), msg_mag)
        agg_phase.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_phase), msg_phase)

        # Normalización por grado: necesaria al subir cutoff a 10 Å, porque el
        # grafo se vuelve mucho más denso y la suma cruda puede explotar.
        deg = torch.zeros((mag.shape[0], 1), dtype=mag.dtype, device=mag.device)
        deg.scatter_add_(0, dst.unsqueeze(1), torch.ones((dst.shape[0], 1), dtype=mag.dtype, device=mag.device))
        deg = deg.clamp_min(1.0)
        agg_mag = agg_mag / deg
        agg_phase = agg_phase / deg

        gate = self.update_gate(torch.cat([mag, agg_mag], dim=-1))
        new_mag = self.norm_mag(mag + gate * agg_mag)
        new_phase = phase + gate * agg_phase

        return ComplexTensor(new_mag, new_phase)


class EdgeBiasProjection(nn.Module):
    """Proyecta RBF a un escalar de bias para atención."""

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, rbf: torch.Tensor) -> torch.Tensor:
        return self.proj(rbf).squeeze(-1)


class ComplexPolarAttention(nn.Module):
    """Atención compleja-polar con bias de aristas basado en RBF."""

    def __init__(self, hidden_dim: int, edge_dim: int = 50):
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

        phase_diff = phase.unsqueeze(1) - phase.unsqueeze(0)
        mag_prod = mag.unsqueeze(1) * mag.unsqueeze(0)
        scores = torch.sum(mag_prod * torch.cos(phase_diff), dim=-1) / self.scale

        if edge_index is not None and rbf is not None and edge_index.numel() > 0 and rbf.numel() > 0:
            bias_vals = self.edge_bias(rbf)
            i_idx, j_idx = edge_index[0], edge_index[1]
            scores = scores.clone()
            scores[i_idx, j_idx] += bias_vals

        attn_weights = F.softmax(scores, dim=1)
        new_mag = torch.matmul(attn_weights, mag)
        new_phase = torch.matmul(attn_weights, phase)

        return ComplexTensor(new_mag, new_phase)


class RealProjection(nn.Module):
    """Proyecta ComplexTensor a reales concatenando real + imag."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim * 2, out_dim)

    def forward(self, cpx: ComplexTensor) -> torch.Tensor:
        z = cpx.as_cartesian()
        return self.lin(torch.cat([z.real, z.imag], dim=-1))
