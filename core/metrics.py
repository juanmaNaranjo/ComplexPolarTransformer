import torch
import torch.nn.functional as F


def _align_shapes(pred: torch.Tensor, target: torch.Tensor):
    """
    Alinea pred y target para evitar broadcasting silencioso.
    Soporta (B,), (B,1), (B,T) y casos donde venga como listas/np.
    """
    pred = torch.as_tensor(pred).float()
    target = torch.as_tensor(target).float()

    # Si ambos son 1D: (B,) ok
    if pred.dim() == 2 and pred.size(-1) == 1:
        pred = pred.view(-1)      # (B,1) -> (B,)
    if target.dim() == 2 and target.size(-1) == 1:
        target = target.view(-1)  # (B,1) -> (B,)

    # Si target viene (B,) pero pred viene (B,T) con T>1 (multi-target),
    # aquí NO lo forzamos: en ese caso debe venir target (B,T) desde el Trainer.
    # Para single-target (T=1) ya quedó (B,).
    return pred, target


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _align_shapes(pred, target)
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _align_shapes(pred, target)
    return torch.sqrt(F.mse_loss(pred, target))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _align_shapes(pred, target)
    return F.mse_loss(pred, target)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred, target = _align_shapes(pred, target)
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-9)


def evaluate_regression(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred, target = _align_shapes(pred, target)
    return {
        "MAE": mae(pred, target).item(),
        "RMSE": rmse(pred, target).item(),
        "MSE": mse(pred, target).item(),
        "R2": r2_score(pred, target).item(),
    }