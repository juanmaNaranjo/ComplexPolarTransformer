"""
Análisis complementarios de PolCT v9 sobre las predicciones ya generadas
(no requiere GPU ni reentrenar):

  1. Distribución del tamaño molecular por partición (train/val/test), para
     verificar la hipótesis de escasez de moléculas pequeñas en entrenamiento.
  2. Figuras: paridad predicción vs. valor real, histograma del error absoluto
     y MAE por tamaño molecular.
  3. Intervalos de confianza bootstrap (95 %) del MAE y del RMSE.

Uso (desde la raíz del proyecto v9):
    python scripts/extra_analysis.py
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"
SIZE_BINS = [(0, 9), (10, 14), (15, 19), (20, 29)]
LE = "\u2264"  # signo "menor o igual" para las etiquetas de la figura


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def atoms_per_dataset_position(sdf_path, csv_len):
    """Replica el filtrado de QM9SDFDataset: misma lista de moléculas válidas."""
    counts = []
    for i, mol in enumerate(Chem.SDMolSupplier(sdf_path, removeHs=False)):
        if mol is None or i >= csv_len:
            continue
        counts.append(mol.GetNumAtoms())
    return np.asarray(counts)


def size_table(n_atoms):
    rows = {}
    for lo, hi in SIZE_BINS:
        k = f"{lo}-{hi}" if lo else f"<={hi}"
        rows[k] = int(((n_atoms >= lo) & (n_atoms <= hi)).sum())
    return rows


def bootstrap_ci(err, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    sample = err[idx]
    mae = np.abs(sample).mean(axis=1)
    rmse = np.sqrt((sample ** 2).mean(axis=1))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"n_boot": n_boot, "seed": seed, "mae_ci95": q(mae), "rmse_ci95": q(rmse),
            "mae_boot_std": float(mae.std(ddof=1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf", default="data/qm9.sdf")
    p.add_argument("--csv", default="data/qm9.csv")
    p.add_argument("--predictions", default="results/predictions_v9_angular_modrelu.csv")
    p.add_argument("--split-file", default="logs_v9_angular_modrelu/split_seed42.json")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out-dir", default="results/extra_analysis")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_len = len(pd.read_csv(args.csv, usecols=[0]))
    n_atoms = atoms_per_dataset_position(args.sdf, csv_len)
    split = json.load(open(args.split_file, encoding="utf-8"))
    by_split = {}
    for name in ("train", "val", "test"):
        n = n_atoms[np.asarray(split[f"{name}_idx"])]
        by_split[name] = {"n": int(len(n)), "bins": size_table(n),
                          "pct_le9": float((n <= 9).mean() * 100),
                          "n_le5": int((n <= 5).sum()), "median_atoms": float(np.median(n))}

    pred = pd.read_csv(args.predictions)
    # idx_eval es la posición dentro del test; idx_original_sdf_csv, el índice en el SDF.
    test_pos = np.asarray(split["test_idx"])
    pred["n_atoms"] = n_atoms[test_pos[pred["idx_eval"].values]]
    err = (pred["y_pred"] - pred["y_true"]).values
    abs_err = np.abs(err)

    mae_by_size = []
    for lo, hi in SIZE_BINS:
        m = (pred["n_atoms"] >= lo) & (pred["n_atoms"] <= hi)
        mae_by_size.append({"bin": f"{lo}-{hi}" if lo else f"<={hi}", "n": int(m.sum()),
                            "mae": float(abs_err[m].mean())})

    boot = bootstrap_ci(err, args.n_boot, seed=42)

    # Figura 1: (a) paridad y (b) error con signo frente al valor real
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    ax.scatter(pred["y_true"], pred["y_pred"], s=4, color=SERIES, alpha=0.35, linewidths=0, rasterized=True)
    lo_v, hi_v = pred["y_true"].min(), pred["y_true"].max()
    ax.plot([lo_v, hi_v], [lo_v, hi_v], color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel(r"$U_0^{\mathrm{atom}}$ real (kcal/mol)")
    ax.set_ylabel(r"$U_0^{\mathrm{atom}}$ predicho (kcal/mol)")
    ax.set_title("(a) Paridad", fontsize=10, color=INK, loc="left")
    ax.text(0.04, 0.95, f"n = {len(pred):,}\nMAE = {abs_err.mean():.3f} kcal/mol",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    style_axes(ax)

    ax2.scatter(pred["y_true"], err, s=4, color=SERIES, alpha=0.35, linewidths=0, rasterized=True)
    ax2.axhline(0, color=INK_2, linewidth=1, linestyle="--")
    ax2.axhspan(-1, 1, color=GRID, alpha=0.5, linewidth=0, zorder=0)
    ax2.set_yscale("symlog", linthresh=2)
    ax2.set_xlabel(r"$U_0^{\mathrm{atom}}$ real (kcal/mol)")
    ax2.set_ylabel("Predicho $-$ real (kcal/mol, symlog)")
    ax2.set_title("(b) Error con signo (banda: $\\pm$1 kcal/mol)", fontsize=10, color=INK, loc="left")
    w = pred.loc[abs_err.argmax()]
    ax2.annotate("H$_2$O (gdb_3)", (w["y_true"], w["y_pred"] - w["y_true"]), xytext=(-70, 8),
                 textcoords="offset points", fontsize=8, color=INK_2,
                 arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.6})
    style_axes(ax2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"paridad_v9.{ext}"), dpi=200)
    plt.close(fig)

    # Figura 2: histograma del error absoluto (eje x logarítmico)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bins = np.logspace(-3, np.log10(abs_err.max()), 60)
    ax.hist(np.clip(abs_err, 1e-3, None), bins=bins, color=SERIES, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, abs_err.max() * 1.3)
    for val, lab, ls in ((np.median(abs_err), f"Mediana ({np.median(abs_err):.2f})", "-"),
                         (abs_err.mean(), f"MAE ({abs_err.mean():.2f})", "--"),
                         (1.0, "Precisión química (1 kcal/mol)", ":")):
        ax.axvline(val, color=INK, linewidth=1, linestyle=ls, label=lab)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    ax.set_xlabel("Error absoluto (kcal/mol, escala logarítmica)")
    ax.set_ylabel("Número de moléculas")
    style_axes(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"histograma_error_v9.{ext}"), dpi=200)
    plt.close(fig)

    # Figura 3: MAE por tamaño molecular
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    labels = [f"{r['bin'].replace('<=', LE)}\n(n = {r['n']:,})" for r in mae_by_size]
    vals = [r["mae"] for r in mae_by_size]
    bars = ax.bar(labels, vals, color=SERIES, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.axhline(abs_err.mean(), color=INK_2, linewidth=0.8, linestyle="--",
               label=f"MAE global ({abs_err.mean():.2f} kcal/mol)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    ax.set_xlabel("Número total de átomos")
    ax.set_ylabel("MAE (kcal/mol)")
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"mae_por_tamano_v9.{ext}"), dpi=200)
    plt.close(fig)

    out = {"size_by_split": by_split, "mae_by_size": mae_by_size,
           "mae": float(abs_err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
           "bootstrap": boot,
           "consistency_check_max_abs_error_matches_csv":
               bool(np.allclose(abs_err, pred["abs_error"].values, atol=1e-3))}
    with open(os.path.join(args.out_dir, "extra_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
