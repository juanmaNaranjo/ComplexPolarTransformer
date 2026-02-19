import torch
import math


class ComplexTensor:
    """
    Representación de un tensor complejo en forma polar:
        z = r * exp(i * θ)

    - r (magnitude): magnitud >= 0
    - θ (phase): fase en radianes

    Diseñado para modelos complejos-polares en aprendizaje profundo.
    """

    def __init__(self, magnitude: torch.Tensor, phase: torch.Tensor):
        """
        Args:
            magnitude (Tensor): |z| >= 0
            phase (Tensor): ángulo en radianes
        """
        self.magnitude = magnitude
        self.phase = self._wrap_phase(phase)

    # =========================
    # Utilidades internas
    # =========================

    def _wrap_phase(self, phase):
        """
        Normaliza la fase al rango [-pi, pi]
        """
        return (phase + math.pi) % (2 * math.pi) - math.pi

    # =========================
    # Conversiones
    # =========================

    def as_cartesian(self):
        """
        Convierte la representación polar a cartesiana (x + iy)
        """
        real = self.magnitude * torch.cos(self.phase)
        imag = self.magnitude * torch.sin(self.phase)
        return torch.complex(real, imag)

    @staticmethod
    def from_cartesian(z: torch.Tensor):
        """
        Crea un ComplexTensor a partir de un tensor complejo cartesiano
        """
        magnitude = torch.abs(z)
        phase = torch.angle(z)
        return ComplexTensor(magnitude, phase)

    # =========================
    # Operaciones complejas
    # =========================

    def add(self, other):
        """
        Suma compleja: z = z1 + z2
        """
        z = self.as_cartesian() + other.as_cartesian()
        return ComplexTensor.from_cartesian(z)

    def multiply(self, other):
        """
        Producto complejo: z = z1 * z2
        En polar: r = r1*r2, θ = θ1+θ2
        """
        magnitude = self.magnitude * other.magnitude
        phase = self.phase + other.phase
        return ComplexTensor(magnitude, phase)

    # =========================
    # Propiedades útiles
    # =========================

    def real(self):
        return self.as_cartesian().real

    def imag(self):
        return self.as_cartesian().imag

    def abs(self):
        return self.magnitude

    def angle(self):
        return self.phase

    # =========================
    # Estabilidad numérica
    # =========================

    def clamp_magnitude(self, min_val=1e-6, max_val=None):
        """
        Evita magnitudes nulas o explosivas
        """
        if max_val is not None:
            mag = torch.clamp(self.magnitude, min=min_val, max=max_val)
        else:
            mag = torch.clamp(self.magnitude, min=min_val)
        return ComplexTensor(mag, self.phase)

    # =========================
    # Debug / Interpretabilidad
    # =========================

    def summary(self):
        """
        Retorna estadísticas simples para análisis interpretativo
        """
        return {
            "mag_mean": self.magnitude.mean().item(),
            "mag_std": self.magnitude.std().item(),
            "phase_mean": self.phase.mean().item(),
            "phase_std": self.phase.std().item(),
        }
