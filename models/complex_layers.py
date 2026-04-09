import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .complex_tensor import ComplexTensor


# ============================================================
# Embedding complejo polar
# (sin cambios — ya estaba correcto)
# ============================================================

class ComplexEmbedding(nn.Module):
    """
    Convierte features reales en representación compleja polar.
    Produce magnitud >= 0 (softplus) y fase en [-pi, pi] (tanh * pi).
    """

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.to_mag = nn.Linear(in_dim, hidden_dim)
        self.to_phase = nn.Linear(in_dim, hidden_dim)

    def forward(self, x):
        magnitude = F.softplus(self.to_mag(x))
        phase = torch.tanh(self.to_phase(x)) * math.pi
        return ComplexTensor(magnitude, phase)


# ============================================================
# Proyección de aristas: distancia → bias de atención
# NUEVO — usa edge_attr para modular la atención entre átomos
# ============================================================

class EdgeBiasProjection(nn.Module):
    """
    Proyecta los atributos de aristas (distancias interatómicas) a un
    escalar de bias por par (i, j), que se suma al score de atención.

    Justificación física: la atención entre dos átomos debe depender
    de su distancia. Átomos lejanos deben recibir menor peso.
    Referencia: Gilmer et al. (2017) Neural Message Passing for QC.

    Args:
        edge_dim: dimensión de edge_attr (típicamente 1 — distancia)
        hidden_dim: dimensión latente del modelo
    """

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        # Proyecta los atributos de arista a un escalar de bias
        self.proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Cutoff suave: penalización aprendible por distancia.
        # Inicializado en -1.0 para que átomos lejanos reciban bias negativo
        # (menor atención), análogo al envelope de SchNet [14].
        # Referencia: Schütt et al. (2018) SchNet.
        self.dist_scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: [E, edge_dim] — atributos de arista.
                       Se asume que edge_attr[:, 0] es la distancia interatómica.
        Returns:
            bias: [E] — un escalar por arista (negativo = menos atención)
        """
        # Bias base desde todos los features
        base_bias = self.proj(edge_attr).squeeze(-1)   # [E]

        # Penalización por distancia: dist_scale * d_ij
        # Con dist_scale < 0, átomos más lejanos reciben penalización mayor
        dist = edge_attr[:, 0].float()   # primera feature = distancia
        distance_penalty = self.dist_scale * dist   # [E]

        return base_bias + distance_penalty   # [E]


# ============================================================
# Atención compleja-polar mejorada
#
# CAMBIO 1: scaling por sqrt(D) — estabiliza softmax
# CAMBIO 3: bias de aristas — modula atención por distancia
# ============================================================

class ComplexPolarAttention(nn.Module):
    """
    Atención compleja basada en magnitud y fase con tres mejoras:

    1. Scaling por sqrt(D):
       El score sin escalar crece con D (dimensión), lo que satura
       el softmax y colapsa los gradientes. Dividir por sqrt(D)
       normaliza la magnitud del producto interno, igual que en
       el Transformer original (Vaswani et al., 2017).

       score_ij = sum(|z_i||z_j| cos(θ_i - θ_j)) / sqrt(D)

    2. (sin cambio) Agregación compleja de magnitud y fase.

    3. Bias de aristas:
       Si se pasa edge_index + edge_bias, el score entre i y j
       se suma con el bias aprendido de su distancia interatómica.
       Esto introduce inductiva física: átomos más cercanos
       reciben mayor atención inicial.
       Referencia: ViSNet (Wang et al., 2023).

    Args:
        hidden_dim: dimensión D del espacio latente
        edge_dim: dimensión de edge_attr (default 1). Si es None,
                  no se usan aristas.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)   # sqrt(D) para scaling

        if edge_dim is not None:
            self.edge_bias = EdgeBiasProjection(edge_dim, hidden_dim)
        else:
            self.edge_bias = None

    def forward(
        self,
        cpx: ComplexTensor,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> ComplexTensor:
        """
        Args:
            cpx:        ComplexTensor [N, D]
            edge_index: [2, E] índices de aristas (opcional)
            edge_attr:  [E, edge_dim] atributos de aristas (opcional)
        Returns:
            ComplexTensor actualizado [N, D]
        """
        mag = cpx.magnitude    # [N, D]
        phase = cpx.phase      # [N, D]
        N = mag.shape[0]

        # --- Score complejo entre todos los pares ---
        # Re(z_i * conj(z_j)) = |z_i||z_j| cos(θ_i - θ_j)
        phase_diff = phase.unsqueeze(1) - phase.unsqueeze(0)   # [N, N, D]
        mag_prod   = mag.unsqueeze(1) * mag.unsqueeze(0)        # [N, N, D]

        # CAMBIO 1: dividir por sqrt(D) antes del softmax
        scores = torch.sum(mag_prod * torch.cos(phase_diff), dim=-1) / self.scale  # [N, N]

        # CAMBIO 3: sumar bias de aristas al score correspondiente
        if self.edge_bias is not None and edge_index is not None and edge_attr is not None:
            bias_vals = self.edge_bias(edge_attr.float())   # [E]
            i_idx, j_idx = edge_index[0], edge_index[1]    # [E] cada uno
            # Añadir bias al score (i→j)
            scores = scores.clone()
            scores[i_idx, j_idx] = scores[i_idx, j_idx] + bias_vals

        # Normalización softmax
        attn_weights = F.softmax(scores, dim=1)   # [N, N]

        # Agregación de magnitud y fase
        new_mag   = torch.matmul(attn_weights, mag)    # [N, D]
        new_phase = torch.matmul(attn_weights, phase)  # [N, D]

        return ComplexTensor(new_mag, new_phase)


# ============================================================
# Proyección complejo → real
# Mejorada: usa parte real E imaginaria (no solo real)
# ============================================================

class RealProjection(nn.Module):
    """
    Proyecta ComplexTensor a valores reales.

    Mejora menor: concatenar parte real e imaginaria antes de
    proyectar en lugar de descartar la imaginaria. Esto preserva
    información de fase en la capa de salida.
    """

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        # dim * 2 porque concatenamos real + imag
        self.lin = nn.Linear(dim * 2, out_dim)

    def forward(self, cpx: ComplexTensor) -> torch.Tensor:
        z = cpx.as_cartesian()
        # Concatenar parte real e imaginaria
        x = torch.cat([z.real, z.imag], dim=-1)   # [N, D*2]
        return self.lin(x)