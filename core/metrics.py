import torch
import torch.nn.functional as F


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (MAE)
    Métrica principal para QM9.
    """
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error (RMSE)
    Penaliza errores grandes.
    """
    return torch.sqrt(F.mse_loss(pred, target))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error (MSE)
    Métrica auxiliar.
    """
    return F.mse_loss(pred, target)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Coeficiente de determinación R²
    Mide la proporción de varianza explicada.
    """
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-9)


def evaluate_regression(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Retorna un diccionario con métricas estándar de regresión.
    Útil para validación y test.
    """
    return {
        "MAE": mae(pred, target).item(),
        "RMSE": rmse(pred, target).item(),
        "MSE": mse(pred, target).item(),
        "R2": r2_score(pred, target).item(),
    }
