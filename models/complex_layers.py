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
        # Usa cpx.magnitude y cpx.phase directamente.
        # Elimina el round-trip polar→cartesiano→polar (4 trig calls por elemento)
        # y los NaN en gradientes de torch.angle(z) cuando |z|→0.
        radius = cpx.magnitude.clamp_min(0.0)
        phase = cpx.phase

        nonzero = (radius > self.eps).to(radius.dtype)
        activated_radius = F.relu(radius + self.bias) * nonzero

        active = activated_radius > 0
        activated_phase = torch.where(active, phase, torch.zeros_like(phase))

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
        # Buffer para que gamma viaje con el modelo al serializar/mover devices.
        self.register_buffer("gamma", torch.tensor(1.0 / (spacing ** 2)))

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


class AngularBasis(nn.Module):
    """
    Expansión angular explícita para tripletas j-i-k.

    Entrada: cos(theta) donde theta es el ángulo entre dos vecinos alrededor
    del mismo átomo central i.
    Salida: gaussianas sobre theta en [0, pi].

    Esta base permite que el mensaje de una arista j->i no dependa solo de la
    distancia d_ij, sino también de la geometría local alrededor de i.
    """

    def __init__(self, num_basis: int = 16):
        super().__init__()
        if num_basis < 1:
            raise ValueError("num_basis debe ser >= 1")
        self.num_basis = int(num_basis)
        centers = torch.linspace(0.0, math.pi, self.num_basis)
        self.register_buffer("centers", centers)
        if self.num_basis == 1:
            spacing = math.pi
        else:
            spacing = math.pi / (self.num_basis - 1)
        self.gamma = 1.0 / (spacing ** 2)

    def forward(self, cos_theta: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.clamp(cos_theta.float(), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta).unsqueeze(-1)
        return torch.exp(-self.gamma * (theta - self.centers) ** 2)


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
    Paso de mensajes complejo ponderado por RBF y, opcionalmente, por contexto angular.

    Base radial:
        msg_ij = phi_rbf(rbf_ij) * z_j

    Variante angular explícita:
        msg_ij = phi_rbf(rbf_ij, A_ij) * z_j

    donde A_ij resume los ángulos j-i-k con otros vecinos k del mismo átomo
    central i. Esto introduce información de tripletas sin cambiar el dataset.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        use_angular: bool = False,
        num_angle_basis: int = 16,
        angular_scale_init: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_angular = bool(use_angular)
        self.num_angle_basis = int(num_angle_basis)

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

        if self.use_angular:
            self.angle_basis = AngularBasis(num_basis=self.num_angle_basis)
            self.angle_to_mag = nn.Sequential(
                nn.Linear(self.num_angle_basis, hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.angle_to_phase = nn.Sequential(
                nn.Linear(self.num_angle_basis, hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            # Escalas aprendibles inicializadas pequeñas para que el modelo arranque
            # cerca del comportamiento radial de v8 y aprenda gradualmente el término angular.
            self.angular_mag_scale = nn.Parameter(torch.tensor(float(angular_scale_init)))
            self.angular_phase_scale = nn.Parameter(torch.tensor(float(angular_scale_init)))
        else:
            self.angle_basis = None
            self.angle_to_mag = None
            self.angle_to_phase = None
            self.angular_mag_scale = None
            self.angular_phase_scale = None

        self.phase_scale = nn.Parameter(torch.tensor(math.pi))
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm_mag = nn.LayerNorm(hidden_dim)

    def _edge_angular_features(self, coords_cart: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Calcula descriptor angular por arista dirigida j→i.

        v10: sort edges by destination (O(E log E)) en lugar de
        torch.nonzero(dst == center) O(N×E) dentro del loop por átomo central.
        Resultado: [E, num_angle_basis].
        """
        src, dst = edge_index[0], edge_index[1]
        e_count = int(edge_index.shape[1])
        device = coords_cart.device
        dtype = coords_cart.dtype
        out = torch.zeros((e_count, self.num_angle_basis), device=device, dtype=dtype)

        if e_count == 0:
            return out

        # Ordenar aristas por destino: O(E log E) único, luego acceso O(m) por centro.
        order = torch.argsort(dst)
        src_sorted = src[order]
        dst_sorted = dst[order]
        unique_centers, counts = torch.unique_consecutive(dst_sorted, return_counts=True)

        edge_offset = 0
        for center, m in zip(unique_centers.tolist(), counts.tolist()):
            m = int(m)
            if m < 2:
                edge_offset += m
                continue

            incoming = order[edge_offset : edge_offset + m]   # índices originales de aristas
            neighbors = src_sorted[edge_offset : edge_offset + m]

            center_coord = coords_cart[int(center)].view(1, 3)
            vec = coords_cart[neighbors] - center_coord        # [m, 3]
            norm = torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)
            vec = vec / norm

            cos_mat = torch.matmul(vec, vec.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # [m, m]
            basis_mat = self.angle_basis(cos_mat.reshape(-1)).reshape(m, m, self.num_angle_basis)

            mask = (~torch.eye(m, dtype=torch.bool, device=device)).to(dtype).unsqueeze(-1)
            denom = float(max(m - 1, 1))
            avg_basis = (basis_mat * mask).sum(dim=1) / denom  # [m, K]
            out[incoming] = avg_basis.to(dtype)

            edge_offset += m

        return out

    def forward(
        self,
        cpx: ComplexTensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        coords_cart: torch.Tensor = None,
    ) -> ComplexTensor:
        if edge_index is None or rbf is None or edge_index.numel() == 0 or rbf.numel() == 0:
            return cpx

        mag = cpx.magnitude
        phase = cpx.phase
        src, dst = edge_index[0], edge_index[1]

        # Construir mensajes en forma polar
        msg_mag = self.edge_to_mag(rbf) * mag[src]
        msg_phase = self.edge_to_phase(rbf) * self.phase_scale + phase[src]

        if self.use_angular and coords_cart is not None:
            angle_feat = self._edge_angular_features(coords_cart.float(), edge_index)
            angle_feat = angle_feat.to(device=mag.device, dtype=mag.dtype)

            angle_mag = self.angle_to_mag(angle_feat)
            angle_phase = self.angle_to_phase(angle_feat)

            msg_mag = msg_mag * (1.0 + self.angular_mag_scale * angle_mag)
            msg_phase = msg_phase + self.angular_phase_scale * math.pi * angle_phase

        # Agregar en espacio CARTESIANO — corrige el promedio aritmético de fases
        # (cantidad circular): suma aritmética de ángulos es incorrecta cerca de ±π.
        msg_real = msg_mag * torch.cos(msg_phase)
        msg_imag = msg_mag * torch.sin(msg_phase)

        agg_real = torch.zeros_like(mag)
        agg_imag = torch.zeros_like(mag)
        agg_real.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_real), msg_real)
        agg_imag.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_imag), msg_imag)

        # Normalizar por grado en espacio cartesiano
        deg = torch.zeros((mag.shape[0], 1), dtype=mag.dtype, device=mag.device)
        deg.scatter_add_(0, dst.unsqueeze(1), torch.ones((dst.shape[0], 1), dtype=mag.dtype, device=mag.device))
        deg = deg.clamp_min(1.0)
        agg_real = agg_real / deg
        agg_imag = agg_imag / deg

        # Magnitud del agregado (para el gate y la norma)
        agg_mag = torch.sqrt(agg_real ** 2 + agg_imag ** 2 + 1e-9)

        # Actualización con gate en espacio cartesiano
        current_real = mag * torch.cos(phase)
        current_imag = mag * torch.sin(phase)

        gate = self.update_gate(torch.cat([mag, agg_mag], dim=-1))
        new_real = current_real + gate * agg_real
        new_imag = current_imag + gate * agg_imag

        # Normalizar magnitud: abs(LayerNorm) preserva gradientes y garantiza no-negatividad.
        # LayerNorm estabiliza la escala; abs refleja valores negativos en lugar de matarlos.
        new_mag_raw = torch.sqrt(new_real ** 2 + new_imag ** 2 + 1e-9)
        new_mag = torch.abs(self.norm_mag(new_mag_raw)).clamp_min(1e-9)
        new_phase = torch.atan2(new_imag, new_real)

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

        # Score = parte real del producto Hermitiano: Re(<z_i, conj(z_j)>)
        phase_diff = phase.unsqueeze(1) - phase.unsqueeze(0)   # [N, N, H]
        mag_prod = mag.unsqueeze(1) * mag.unsqueeze(0)          # [N, N, H]
        scores = torch.sum(mag_prod * torch.cos(phase_diff), dim=-1) / self.scale  # [N, N]

        if edge_index is not None and rbf is not None and edge_index.numel() > 0 and rbf.numel() > 0:
            bias_vals = self.edge_bias(rbf)
            i_idx, j_idx = edge_index[0], edge_index[1]
            scores = scores.clone()
            scores[i_idx, j_idx] = scores[i_idx, j_idx] + bias_vals

        attn_weights = F.softmax(scores, dim=1)   # [N, N]

        # Agregar en espacio CARTESIANO — corrige el promedio aritmético de fases.
        # matmul(attn, phase) es incorrecto para cantidades circulares cerca de ±π.
        real = mag * torch.cos(phase)                     # [N, H]
        imag = mag * torch.sin(phase)                     # [N, H]
        new_real = torch.matmul(attn_weights, real)       # [N, H]
        new_imag = torch.matmul(attn_weights, imag)       # [N, H]

        new_mag = torch.sqrt(new_real ** 2 + new_imag ** 2 + 1e-9)
        new_phase = torch.atan2(new_imag, new_real)

        return ComplexTensor(new_mag, new_phase)


class RealProjection(nn.Module):
    """Proyecta ComplexTensor a reales concatenando real + imag."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim * 2, out_dim)

    def forward(self, cpx: ComplexTensor) -> torch.Tensor:
        # Calcula real/imag directamente sin crear un tensor complejo intermedio.
        real = cpx.magnitude * torch.cos(cpx.phase)
        imag = cpx.magnitude * torch.sin(cpx.phase)
        return self.lin(torch.cat([real, imag], dim=-1))
