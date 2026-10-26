import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def find_column(df, candidates):
    """
    Busca una columna posible dentro del CSV de predicciones.
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No se encontró ninguna columna válida entre: {candidates}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa MAE, RMSE y R2 en 5 subconjuntos aleatorios del test sin reentrenar."
    )

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Ruta al CSV de predicciones del modelo sobre el test original."
    )

    parser.add_argument(
        "--n_subsets",
        type=int,
        default=5,
        help="Número de subconjuntos aleatorios."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/subset_eval_v9",
        help="Carpeta de salida."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.predictions)

    # Ajusta automáticamente nombres comunes de columnas
    y_true_col = find_column(
        df,
        ["y_true", "target", "real", "valores_reales", "true", "actual"]
    )

    y_pred_col = find_column(
        df,
        ["y_pred", "prediction", "pred", "predicciones", "predicted"]
    )

    y_true = df[y_true_col].to_numpy(dtype=float)
    y_pred = df[y_pred_col].to_numpy(dtype=float)

    n = len(df)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    # Divide el test en 5 subconjuntos disjuntos
    subsets = np.array_split(indices, args.n_subsets)

    results = []

    for i, subset_idx in enumerate(subsets, start=1):
        yt = y_true[subset_idx]
        yp = y_pred[subset_idx]

        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        r2 = r2_score(yt, yp)

        result = {
            "subset": i,
            "n_samples": int(len(subset_idx)),
            "mae_kcal_mol": float(mae),
            "rmse_kcal_mol": float(rmse),
            "r2": float(r2)
        }

        results.append(result)

        subset_df = df.iloc[subset_idx].copy()
        subset_df.to_csv(
            os.path.join(args.output_dir, f"subset_{i}.csv"),
            index=False
        )

    results_df = pd.DataFrame(results)

    summary = {
        "model": "ComplexPolarTransformer v9 angular explicito + modReLU",
        "predictions_file": args.predictions,
        "test_samples_total": int(n),
        "n_subsets": args.n_subsets,
        "seed": args.seed,
        "subset_strategy": "random_disjoint_partition_of_original_test",
        "metrics_by_subset": results,
        "summary": {
            "mae_mean": float(results_df["mae_kcal_mol"].mean()),
            "mae_std": float(results_df["mae_kcal_mol"].std(ddof=1)),
            "rmse_mean": float(results_df["rmse_kcal_mol"].mean()),
            "rmse_std": float(results_df["rmse_kcal_mol"].std(ddof=1)),
            "r2_mean": float(results_df["r2"].mean()),
            "r2_std": float(results_df["r2"].std(ddof=1))
        }
    }

    results_df.to_csv(
        os.path.join(args.output_dir, "subset_metrics.csv"),
        index=False
    )

    with open(os.path.join(args.output_dir, "subset_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n=== Métricas por subconjunto ===")
    print(results_df.to_string(index=False))

    print("\n=== Resumen ===")
    print(f"MAE  promedio:  {summary['summary']['mae_mean']:.6f} ± {summary['summary']['mae_std']:.6f} kcal/mol")
    print(f"RMSE promedio:  {summary['summary']['rmse_mean']:.6f} ± {summary['summary']['rmse_std']:.6f} kcal/mol")
    print(f"R2   promedio:  {summary['summary']['r2_mean']:.8f} ± {summary['summary']['r2_std']:.8f}")

    print(f"\nResultados guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()