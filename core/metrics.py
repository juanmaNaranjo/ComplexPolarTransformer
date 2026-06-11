"""
core/metrics.py
 
Funciones de evaluación para regresión molecular.
Diseñado para ser robusto ante NaN/Inf en predicciones desnormalizadas.
"""
 
import torch
import math
 
 
def evaluate_regression(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Calcula MAE, RMSE y R² de forma robusta.
 
    Args:
        preds:   Tensor [N, 1] o [N] — predicciones desnormalizadas
        targets: Tensor [N, 1] o [N] — targets desnormalizados
 
    Returns:
        dict con claves: mae, rmse, r2
        Si hay NaN/Inf, los filtra antes de calcular.
    """
    # Aplanar a [N]
    preds   = preds.detach().float().view(-1)
    targets = targets.detach().float().view(-1)
 
    # Filtrar NaN e Inf — si existen, el modelo tiene un problema upstream
    mask = torch.isfinite(preds) & torch.isfinite(targets)
    if mask.sum() == 0:
        # No hay ningún valor válido — reportar NaN con advertencia
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
 
    if mask.sum() < len(mask):
        n_bad = len(mask) - mask.sum().item()
        print(f"[METRICS WARNING] {n_bad} muestras con NaN/Inf filtradas de {len(mask)} totales")
 
    p = preds[mask]
    t = targets[mask]
 
    # MAE
    mae = (p - t).abs().mean().item()
 
    # RMSE
    rmse = math.sqrt(((p - t) ** 2).mean().item())
 
    # R² = 1 - SS_res / SS_tot
    ss_res = ((p - t) ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item()
 
    if ss_tot < 1e-12:
        # Todos los targets son iguales — R² indefinido
        r2 = float("nan")
    else:
        r2 = 1.0 - ss_res / ss_tot
 
    return {"mae": mae, "rmse": rmse, "r2": r2}