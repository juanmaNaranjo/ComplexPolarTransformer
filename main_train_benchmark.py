import argparse
import json
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from core.dataset import QM9SDFDataset
from core.trainer_benchmark import Trainer
from core.utils import set_seed
from models.complex_model_beta import ComplexPolarTransformerBeta


def collate_mol(batch):
    """Collate para moléculas de tamaño variable."""
    return {
        "coords_cart": [item["coords_cart"] for item in batch],
        "coords_spherical": [item["coords_spherical"] for item in batch],
        "atom_types": [item["atom_types"] for item in batch],
        "edge_index": [item["edge_index"] for item in batch],
        "edge_attr": [item["edge_attr"] for item in batch],
        "y": torch.stack([item["y"] for item in batch]),
        "num_atoms": torch.tensor([item["num_atoms"] for item in batch], dtype=torch.float32),
        "num_edges": torch.tensor([item["num_edges"] for item in batch], dtype=torch.long),
        "original_idx": torch.tensor([item.get("original_idx", -1) for item in batch], dtype=torch.long),
    }


def make_splits(n_total: int, seed: int, benchmark_split: bool = True):
    """
    Crea índices train/val/test.
    - benchmark_split=True: usa 110k/10k/rest si n_total >= 120k.
    - benchmark_split=False: usa 80/10/10.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()

    if benchmark_split and n_total >= 120_000:
        n_train = 110_000
        n_val = 10_000
        n_test = n_total - n_train - n_val
        if n_test <= 0:
            benchmark_split = False

    if not benchmark_split:
        n_train = int(0.80 * n_total)
        n_val = int(0.10 * n_total)
        n_test = n_total - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx, benchmark_split


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_mol,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento benchmark ComplexPolarTransformer")
    parser.add_argument(
        "--config",
        default="experiments/beta_train_benchmark.yaml",
        help="Ruta del YAML de configuración",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    target = dataset_cfg.get("target")
    targets_list = dataset_cfg.get("targets")
    if target is None:
        if isinstance(targets_list, list) and len(targets_list) == 1:
            target = targets_list[0]
        elif isinstance(targets_list, list) and len(targets_list) > 1:
            raise ValueError(
                "Para benchmark single-target define dataset.target en el YAML. "
                "dataset.targets contiene múltiples valores."
            )
        else:
            target = "u0"

    cutoff = float(model_cfg.get("cutoff", 5.0))
    per_atom_norm = bool(cfg.get("per_atom_norm", True))
    normalize_target = bool(cfg.get("normalize_target", True))

    print(f"[INFO] Target a predecir: {target}")
    print(f"[INFO] Cutoff/dataset max_radius: {cutoff:.3f} Å")
    print(f"[INFO] normalize_target={normalize_target} | per_atom_norm={per_atom_norm}")

    dataset = QM9SDFDataset(
        sdf_path=dataset_cfg["sdf"],
        csv_path=dataset_cfg["csv"],
        target_col=target,
        max_radius=cutoff,
    )

    sample_size = cfg.get("sample_size", None)
    sample_indices = None
    if sample_size is not None:
        random.seed(seed)
        total = len(dataset)
        sample_size = min(int(sample_size), total)
        sample_indices = random.sample(range(total), sample_size)
        dataset = Subset(dataset, sample_indices)
        print(f"[INFO] Usando {sample_size} muestras aleatorias de {total} disponibles.")

    benchmark_split = bool(cfg.get("benchmark_split", True))
    n_total = len(dataset)
    train_idx, val_idx, test_idx, benchmark_split = make_splits(
        n_total=n_total,
        seed=seed,
        benchmark_split=benchmark_split,
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    split_path = cfg.get("split_path", os.path.join("logs", f"split_seed{seed}.json"))
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "benchmark_split": benchmark_split,
                "n_total": n_total,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_test": len(test_ds),
                "target": target,
                "cutoff": cutoff,
                "max_radius": cutoff,
                "num_rbf": int(model_cfg.get("num_rbf", 50)),
                "use_angular": bool(model_cfg.get("use_angular", False)),
                "num_angle_basis": int(model_cfg.get("num_angle_basis", 16)),
                "per_atom_norm": per_atom_norm,
                "normalize_target": normalize_target,
                "sample_size": sample_size,
                "sample_indices": sample_indices,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[INFO] Split guardado en: {split_path}")

    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = torch.cuda.is_available()
    val_workers = max(0, num_workers // 2)

    train_dl = build_dataloader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = build_dataloader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
    )
    test_dl = build_dataloader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=pin_memory,
    )

    model = ComplexPolarTransformerBeta(
        in_dim=int(model_cfg["in_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg.get("out_dim", 1)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 1)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_rbf=int(model_cfg.get("num_rbf", 64)),
        cutoff=cutoff,
        edge_dim=int(model_cfg.get("edge_dim", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_residuals=bool(model_cfg.get("use_residuals", False)),
        use_layernorm=bool(model_cfg.get("use_layernorm", False)),
        activation=str(model_cfg.get("activation", "modrelu")),
        modrelu_init_bias=float(model_cfg.get("modrelu_init_bias", -0.1)),
        modrelu_eps=float(model_cfg.get("modrelu_eps", 1e-8)),
        use_angular=bool(model_cfg.get("use_angular", False)),
        num_angle_basis=int(model_cfg.get("num_angle_basis", 16)),
        angular_scale_init=float(model_cfg.get("angular_scale_init", 0.1)),
    )

    early    = cfg.get("early_stopping", {}) or {}
    sched    = cfg.get("scheduler",      None)
    ema_cfg  = cfg.get("ema",            {}) or {}
    amp_cfg  = cfg.get("amp",            {}) or {}

    hparams = {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "learning_rate": float(cfg.get("learning_rate")),
        "batch_size": int(cfg.get("batch_size")),
        "max_epochs": int(cfg.get("max_epochs")),
        "seed": seed,
        "num_workers": num_workers,
        "target": target,
        "split_path": split_path,
        "benchmark_split": benchmark_split,
        "cutoff": cutoff,
        "max_radius": cutoff,
        "per_atom_norm": per_atom_norm,
        "normalize_target": normalize_target,
    }

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        max_epochs=int(cfg["max_epochs"]),
        ckpt_dir=cfg.get("ckpt_dir", "checkpoints"),
        log_dir=cfg.get("log_dir", "logs"),
        normalize_target=normalize_target,
        per_atom_norm=per_atom_norm,
        hparams=hparams,
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        patience=int(early.get("patience", 30)),
        min_delta=float(early.get("min_delta", 5e-4)),
        scheduler_cfg=sched,
        use_ema=bool(ema_cfg.get("enabled", True)),
        ema_decay=float(ema_cfg.get("decay", 0.999)),
        use_amp=bool(amp_cfg.get("enabled", True)),
        phase_reg_weight=float(cfg.get("phase_reg_weight", 0.0)),
    )

    trainer.fit()
