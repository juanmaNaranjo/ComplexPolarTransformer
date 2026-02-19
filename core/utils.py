import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Fija semillas para reproducibilidad en Python, NumPy y PyTorch.

    Args:
        seed (int): valor de la semilla
        deterministic (bool): fuerza comportamiento determinista en CUDA
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[Seed] Experimento inicializado con seed = {seed}")


def count_parameters(model) -> int:
    """
    Cuenta parámetros entrenables del modelo.
    Útil para reportes y comparación de complejidad.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    """
    Retorna un resumen simple del modelo para logging.
    """
    n_params = count_parameters(model)
    return {
        "num_parameters": n_params,
        "num_parameters_M": n_params / 1e6,
    }
