import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_tensor import ComplexTensor


class ShiftedSoftplus(nn.Module):
    """
    SSP(x) = softplus(x) - log(2).
    Activación suave de SchNet: continua, centrada cerca de 0, gradiente suave.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - math.log(2.0)


class MagnitudeRMSNorm(nn.Module):
    """
    RMS normalization para magnitudes complejas.

    Reemplaza LayerNorm + abs() que tenía dos problemas:
    - LayerNorm produce valores negativos → abs() introduce subgradiente en 0.
    - El subgradiente discontinuo causa neuronas muertas cuando mag≈0.

    RMSNorm preserva no-negatividad sin discontinuidades:
        x_norm = x / sqrt(mean(x²) + ε) * weight
    El clamp_min(0) al final garantiza no-negatividad para magnitudes.
    weight es learnable inicializado en 1 (misma capacidad que LayerNorm sin bias).
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).clamp_min(0.0)


class ModReLU(nn.Module):
    """
    modReLU(z; b_eff) = ReLU(|z| + b_eff) · e^{iθ}, si |z| > eps; 0 si no.
    Conserva la fase exactamente en la región activa (|z| > -b_eff).

    Fix de amplificación:
        b_eff = -softplus(-self.bias) ≤ 0  siempre.

    El parámetro bruto `self.bias` es libre en ℝ; la proyección via softplus
    garantiza que b_eff nunca sea positivo, por lo que ModReLU actúa siempre
    como puerta selectiva (threshold) y nunca como amplificador.

    Inicialización: init_bias alto (ej. 5.0) → b_eff ≈ -0.007 (near-identity).
                    init_bias bajo (ej. -5.0) → b_eff ≈ -5.0 (fuerte poda).
    La función es C∞ en self.bias, sin kinks ni gradientes nulos en 0.
    """
    def __init__(self, hidden_dim: int, init_bias: float = 5.0, eps: float = 1e-8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.bias = nn.Parameter(torch.full((self.hidden_dim,), float(init_bias)))

    def forward(self, cpx: ComplexTensor) -> ComplexTensor:
        radius = cpx.magnitude.clamp_min(0.0)
        phase  = cpx.phase
        # Umbral efectivo ≤ 0: softplus garantiza que el bias nunca amplifique
        effective_bias   = -F.softplus(-self.bias)                  # [H], ≤ 0
        nonzero          = (radius > self.eps).to(radius.dtype)
        activated_radius = F.relu(radius + effective_bias) * nonzero
        active           = activated_radius > 0
        activated_phase  = torch.where(active, phase, torch.zeros_like(phase))
        return ComplexTensor(activated_radius, activated_phase)


class RBFExpansion(nn.Module):
    """
    Expansión radial gaussiana con cosine cutoff para distancias interatómicas.
    edge_attr[:, 0] es la distancia real en Å.

        rbf_k(d) = exp(-γ(d - μ_k)²) · f_cut(d)
        f_cut(d) = 0.5(cos(π·d/c) + 1)  si d < c, else 0

    Centros uniformes en [0, cutoff]; gamma como buffer (serializable, device-safe).
    """
    def __init__(self, num_rbf: int = 50, cutoff: float = 5.0):
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf debe ser >= 1")
        if cutoff <= 0:
            raise ValueError("cutoff debe ser > 0")
        self.num_rbf = int(num_rbf)
        self.cutoff  = float(cutoff)
        centers = torch.linspace(0.0, self.cutoff, self.num_rbf)
        self.register_buffer("centers", centers)
        spacing = self.cutoff / (self.num_rbf - 1) if self.num_rbf > 1 else self.cutoff
        self.register_buffer("gamma", torch.tensor(1.0 / (spacing ** 2)))

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr no puede ser None en RBFExpansion.forward")
        if edge_attr.numel() == 0:
            return edge_attr.new_empty((0, self.num_rbf))
        dist = edge_attr[:, 0].float().unsqueeze(-1)           # [E, 1]
        rbf  = torch.exp(-self.gamma * (dist - self.centers) ** 2)
        inside     = (dist < self.cutoff).float()
        cos_cutoff = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * inside
        return rbf * cos_cutoff                                 # [E, num_rbf]


class AngularBasis(nn.Module):
    """
    Expansión angular gaussiana para tripletas j-i-k.
    Entrada: cos(θ_jik), salida: base gaussiana sobre θ ∈ [0, π].

    Los ángulos entre pares de vecinos son invariantes rotacionales (dot product
    de vectores de enlace normalizados). Esto sí aporta geometría sin romper
    la invarianza.

    gamma ahora es buffer — corrección de bug que causaba fallo en GPU con float16
    (antes era float Python, ignorado por .to(device/dtype)).
    """
    def __init__(self, num_basis: int = 16):
        super().__init__()
        if num_basis < 1:
            raise ValueError("num_basis debe ser >= 1")
        self.num_basis = int(num_basis)
        centers = torch.linspace(0.0, math.pi, self.num_basis)
        self.register_buffer("centers", centers)
        spacing = math.pi / (self.num_basis - 1) if self.num_basis > 1 else math.pi
        self.register_buffer("gamma", torch.tensor(1.0 / (spacing ** 2)))

    def forward(self, cos_theta: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.clamp(cos_theta.float(), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta).unsqueeze(-1)            # [*, 1]
        return torch.exp(-self.gamma * (theta - self.centers) ** 2)


class ComplexEmbedding(nn.Module):
    """
    Embedding complejo profundo para features atómicas físicas.

    MLPs de 2 capas (vs. 1 capa anterior) dan capacidad adicional para
    procesar las 12 features enriquecidas (hibridación, aromaticidad, etc.).

    Magnitud: softplus(MLP(x)) ≥ 0 — intensidad atómica.
    Fase:     tanh(MLP(x))·π ∈ (-π, π) — orientación/polaridad atómica.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        mid = max(hidden_dim, in_dim * 2)
        self.to_mag = nn.Sequential(
            nn.Linear(in_dim, mid),
            ShiftedSoftplus(),
            nn.Linear(mid, hidden_dim),
            nn.Softplus(),
        )
        self.to_phase = nn.Sequential(
            nn.Linear(in_dim, mid),
            ShiftedSoftplus(),
            nn.Linear(mid, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> ComplexTensor:
        magnitude = self.to_mag(x)
        phase     = self.to_phase(x) * math.pi
        return ComplexTensor(magnitude, phase)


class ComplexPolarAttention(nn.Module):
    """
    Multi-head complex polar attention con proyecciones Q/K/V y edge-masking.

    Mejoras sobre V1 (auditoría):

    1. Score asimétrico con Q/K/V separados:
       score(i,j) = Re(Q_i · conj(K_j)) / √head_dim
       Q = W_Q(z), K = W_K(z) — proyecciones independientes.
       Antes: score(i,j) = Re(z_i · conj(z_j)) — simétrico, menos expresivo.

    2. Edge-masking geométrico:
       Pares (i,j) sin arista reciben score = -∞ → attn = 0.
       Garantiza que la atención sea local (dentro del cutoff).
       Cada átomo siempre puede atenderse a sí mismo (diagonal = 0).

    3. Multi-head (num_heads cabezas):
       Cada cabeza puede capturar tipo de interacción diferente
       (radial, angular, electrónico, estérico).

    4. Agregación V en espacio cartesiano (preservado de V1):
       Corrige el circular mean problem de las fases.

    5. Proyección de salida W_O aplicada igual a real e imag → complex-linear.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, edge_dim: int = 50):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} debe ser divisible por num_heads={num_heads}"
            )
        self.num_heads  = int(num_heads)
        self.head_dim   = hidden_dim // num_heads
        self.hidden_dim = int(hidden_dim)
        self.scale      = math.sqrt(self.head_dim)

        # Proyecciones Q, K independientes — asimetría del score
        self.W_Q_mag   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K_mag   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_Q_phase = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K_phase = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Proyección de valor V
        self.W_V_mag   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V_phase = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Proyección de salida
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        # Bias de arista por cabeza: edge_dim → num_heads escalares
        mid_bias = max(edge_dim // 2, num_heads * 2)
        self.edge_bias = nn.Sequential(
            nn.Linear(edge_dim, mid_bias),
            ShiftedSoftplus(),
            nn.Linear(mid_bias, num_heads),
        )

    def forward(
        self,
        cpx:        ComplexTensor,
        edge_index: torch.Tensor = None,
        rbf:        torch.Tensor = None,
    ) -> ComplexTensor:
        mag, phase = cpx.magnitude, cpx.phase      # [N, H]
        N  = mag.shape[0]
        nh = self.num_heads
        hd = self.head_dim

        # ── Proyectar Q, K, V en espacio polar ───────────────────────────
        q_mag   = F.softplus(self.W_Q_mag(mag)).view(N, nh, hd)             # [N,nh,hd]
        k_mag   = F.softplus(self.W_K_mag(mag)).view(N, nh, hd)
        q_phase = (torch.tanh(self.W_Q_phase(phase)) * math.pi).view(N, nh, hd)
        k_phase = (torch.tanh(self.W_K_phase(phase)) * math.pi).view(N, nh, hd)
        v_mag   = F.softplus(self.W_V_mag(mag)).view(N, nh, hd)
        v_phase = (torch.tanh(self.W_V_phase(phase)) * math.pi).view(N, nh, hd)

        # ── Score Hermítico asimétrico: Re(Q_i·conj(K_j))/√hd ───────────
        # Re(Q·K†) = q_real·k_real + q_imag·k_imag  (por componentes)
        # Implementación con einsum evita materializar [N,N,nh,hd]:
        #   Antes: 2×[N,N,nh,hd] ≈ 660 KB/capa/molécula → presión de memoria GPU.
        #   Ahora:  [N,nh,hd] inputs + [N,N,nh] output ≈ 41 KB → 16× menos.
        q_real = q_mag * torch.cos(q_phase)   # [N, nh, hd]
        q_imag = q_mag * torch.sin(q_phase)
        k_real = k_mag * torch.cos(k_phase)
        k_imag = k_mag * torch.sin(k_phase)
        scores = (
            torch.einsum("ihd,jhd->ijh", q_real, k_real) +
            torch.einsum("ihd,jhd->ijh", q_imag, k_imag)
        ) / self.scale                                             # [N, N, nh]

        # ── Edge-masking ─────────────────────────────────────────────────
        if edge_index is not None and edge_index.numel() > 0:
            i_idx, j_idx = edge_index[0], edge_index[1]

            # Pares sin arista = -inf; pares con arista = 0 (sin penalización extra)
            attn_mask = scores.new_full((N, N, nh), float("-inf"))
            attn_mask[i_idx, j_idx, :] = 0.0
            # Cada átomo puede atenderse a sí mismo
            diag = torch.arange(N, device=mag.device)
            attn_mask[diag, diag, :] = 0.0
            scores = scores + attn_mask

            # Bias de arista condicionado en RBF (diferente por cabeza)
            if rbf is not None and rbf.numel() > 0:
                bias = self.edge_bias(rbf)                    # [E, nh]
                bias_full = scores.new_zeros(N, N, nh)
                # AMP: torch.cos() produce float32 aunque la entrada sea float16,
                # por lo que scores (y bias_full) quedan en float32, pero Linear
                # (edge_bias) produce float16. Cast explícito evita el RuntimeError.
                bias_full[i_idx, j_idx, :] = bias.to(bias_full.dtype)
                scores = scores + bias_full

        attn_weights = F.softmax(scores, dim=1)               # [N,N,nh]

        # ── Agregar V en espacio cartesiano (corrige circular mean) ──────
        v_real = v_mag * torch.cos(v_phase)                   # [N,nh,hd]
        v_imag = v_mag * torch.sin(v_phase)

        # [nh,N,N] × [nh,N,hd] → [nh,N,hd] → [N,H]
        aw       = attn_weights.permute(2, 0, 1)              # [nh,N,N]
        vr       = v_real.permute(1, 0, 2)                    # [nh,N,hd]
        vi       = v_imag.permute(1, 0, 2)
        new_real = torch.bmm(aw, vr).permute(1, 0, 2).reshape(N, self.hidden_dim)
        new_imag = torch.bmm(aw, vi).permute(1, 0, 2).reshape(N, self.hidden_dim)

        # Proyección de salida (mismos pesos para real e imag → complex-linear)
        new_real = self.W_O(new_real)
        new_imag = self.W_O(new_imag)

        new_mag   = torch.sqrt(new_real ** 2 + new_imag ** 2 + 1e-9)
        new_phase = torch.atan2(new_imag, new_real)
        return ComplexTensor(new_mag, new_phase)


class ComplexMessagePassing(nn.Module):
    """
    Paso de mensajes complejo ponderado por RBF y, opcionalmente, contexto angular.

    Base radial:
        msg_ij = φ_rbf(rbf_ij) * z_j

    Variante angular explícita:
        msg_ij = φ_rbf(rbf_ij) * z_j  modulado por A_ij (ángulos j-i-k invariantes)

    Agregación en espacio CARTESIANO — corrige circular mean de fases.
    Actualización con gate y RMS normalization en magnitudes.
    """

    def __init__(
        self,
        hidden_dim:       int,
        edge_dim:         int,
        use_angular:      bool  = False,
        num_angle_basis:  int   = 16,
        angular_scale_init: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim    = int(hidden_dim)
        self.use_angular   = bool(use_angular)
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
            self.angle_basis   = AngularBasis(num_basis=self.num_angle_basis)
            self.angle_to_mag  = nn.Sequential(
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
            self.angular_mag_scale   = nn.Parameter(torch.tensor(float(angular_scale_init)))
            self.angular_phase_scale = nn.Parameter(torch.tensor(float(angular_scale_init)))
        else:
            self.angle_basis         = None
            self.angle_to_mag        = None
            self.angle_to_phase      = None
            self.angular_mag_scale   = None
            self.angular_phase_scale = None

        self.phase_scale  = nn.Parameter(torch.tensor(math.pi))
        self.update_gate  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.norm_mag = MagnitudeRMSNorm(hidden_dim)

    def _edge_angular_features(
        self,
        coords_cart: torch.Tensor,
        edge_index:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Descriptor angular por arista dirigida j→i.
        O(E log E) — sort por destino, luego acceso O(m) por centro.
        Los ángulos cos(θ_jik) son invariantes rotacionales.
        """
        src, dst = edge_index[0], edge_index[1]
        e_count  = int(edge_index.shape[1])
        device   = coords_cart.device
        dtype    = coords_cart.dtype
        out = torch.zeros((e_count, self.num_angle_basis), device=device, dtype=dtype)

        if e_count == 0:
            return out

        order      = torch.argsort(dst)
        src_sorted = src[order]
        dst_sorted = dst[order]
        unique_centers, counts = torch.unique_consecutive(dst_sorted, return_counts=True)

        edge_offset = 0
        for center, m in zip(unique_centers.tolist(), counts.tolist()):
            m = int(m)
            if m < 2:
                edge_offset += m
                continue
            incoming  = order[edge_offset : edge_offset + m]
            neighbors = src_sorted[edge_offset : edge_offset + m]

            center_coord = coords_cart[int(center)].view(1, 3)
            vec  = coords_cart[neighbors] - center_coord              # [m, 3]
            norm = torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-8)
            vec  = vec / norm

            cos_mat   = torch.matmul(vec, vec.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            basis_mat = self.angle_basis(cos_mat.reshape(-1)).reshape(m, m, self.num_angle_basis)
            mask      = (~torch.eye(m, dtype=torch.bool, device=device)).to(dtype).unsqueeze(-1)
            denom     = float(max(m - 1, 1))
            avg_basis = (basis_mat * mask).sum(dim=1) / denom         # [m, K]
            out[incoming] = avg_basis.to(dtype)
            edge_offset += m

        return out

    def forward(
        self,
        cpx:                   ComplexTensor,
        edge_index:            torch.Tensor,
        rbf:                   torch.Tensor,
        coords_cart:           torch.Tensor = None,
        precomputed_angle_feat: torch.Tensor = None,
    ) -> ComplexTensor:
        """
        precomputed_angle_feat: features angulares ya calculadas externamente [E, K].
        Si se pasa, se usa directamente (evita recalcular por capa).
        Si es None y use_angular=True, se recalcula a partir de coords_cart.
        """
        if edge_index is None or rbf is None or edge_index.numel() == 0 or rbf.numel() == 0:
            return cpx

        mag, phase = cpx.magnitude, cpx.phase
        src, dst   = edge_index[0], edge_index[1]

        # Construir mensajes en forma polar
        msg_mag   = self.edge_to_mag(rbf) * mag[src]
        msg_phase = self.edge_to_phase(rbf) * self.phase_scale + phase[src]

        if self.use_angular:
            # Usar features precomputadas si están disponibles (evita bucle Python por capa)
            if precomputed_angle_feat is not None:
                angle_feat = precomputed_angle_feat
            elif coords_cart is not None:
                angle_feat = self._edge_angular_features(coords_cart.float(), edge_index)
                angle_feat = angle_feat.to(device=mag.device, dtype=mag.dtype)
            else:
                angle_feat = None

            if angle_feat is not None:
                angle_mag   = self.angle_to_mag(angle_feat)
                angle_phase = self.angle_to_phase(angle_feat)
                msg_mag   = msg_mag   * (1.0 + self.angular_mag_scale   * angle_mag)
                msg_phase = msg_phase + self.angular_phase_scale * math.pi * angle_phase

        # Agregar en espacio CARTESIANO — corrige circular mean de fases
        msg_real = msg_mag * torch.cos(msg_phase)
        msg_imag = msg_mag * torch.sin(msg_phase)

        agg_real = torch.zeros_like(mag)
        agg_imag = torch.zeros_like(mag)
        agg_real.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_real), msg_real)
        agg_imag.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_imag), msg_imag)

        deg = torch.zeros((mag.shape[0], 1), dtype=mag.dtype, device=mag.device)
        deg.scatter_add_(0, dst.unsqueeze(1),
                         torch.ones((dst.shape[0], 1), dtype=mag.dtype, device=mag.device))
        deg = deg.clamp_min(1.0)
        agg_real = agg_real / deg
        agg_imag = agg_imag / deg

        agg_mag = torch.sqrt(agg_real ** 2 + agg_imag ** 2 + 1e-9)

        # Actualización con gate en espacio cartesiano
        current_real = mag * torch.cos(phase)
        current_imag = mag * torch.sin(phase)
        gate     = self.update_gate(torch.cat([mag, agg_mag], dim=-1))
        new_real = current_real + gate * agg_real
        new_imag = current_imag + gate * agg_imag

        # RMSNorm sobre magnitudes (sin abs() — preserva gradientes)
        new_mag_raw = torch.sqrt(new_real ** 2 + new_imag ** 2 + 1e-9)
        new_mag     = self.norm_mag(new_mag_raw).clamp_min(1e-9)
        new_phase   = torch.atan2(new_imag, new_real)

        return ComplexTensor(new_mag, new_phase)


class RealProjection(nn.Module):
    """Proyecta ComplexTensor a reales concatenando real + imag."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim * 2, out_dim)

    def forward(self, cpx: ComplexTensor) -> torch.Tensor:
        real = cpx.magnitude * torch.cos(cpx.phase)
        imag = cpx.magnitude * torch.sin(cpx.phase)
        return self.lin(torch.cat([real, imag], dim=-1))
