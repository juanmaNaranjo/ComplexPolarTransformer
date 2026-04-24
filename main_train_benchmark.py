import os
import json
import yaml
import torch
import random
from torch.utils.data import DataLoader, Subset
from models.complex_model_beta import ComplexPolarTransformerBeta
from core.dataset import QM9SDFDataset
from core.trainer_benchmark import Trainer
from core.utils import set_seed


def collate_mol(batch):
    """
    Collate function para DataLoader que maneja batches de moléculas de distinto tamaño.
    Retorna un diccionario de tensores/lists.
    """
    return {
        "coords_cart": [item["coords_cart"] for item in batch],
        "coords_spherical": [item["coords_spherical"] for item in batch],
        "atom_types": [item["atom_types"] for item in batch],
        "edge_index": [item["edge_index"] for item in batch],
        "edge_attr": [item["edge_attr"] for item in batch],
        "y": torch.stack([item["y"] for item in batch]),
    }


def make_splits(n_total: int, seed: int, benchmark_split: bool = True):
    """
    Crea índices train/val/test.
    - benchmark_split=True: intenta usar 110k/10k/rest (protocolo común en QM9)
    - benchmark_split=False: usa 80/10/10
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()

    if benchmark_split and n_total >= 120_000:
        n_train = 110_000
        n_val = 10_000
        n_test = n_total - n_train - n_val
        if n_test <= 0:
            # fallback
            benchmark_split = False

    if not benchmark_split:
        n_train = int(0.80 * n_total)
        n_val = int(0.10 * n_total)
        n_test = n_total - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


if __name__ == "__main__":

    # ============================
    # 1) Cargar YAML
    # ============================
    cfg_path = "experiments/beta_train_benchmark.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg["dataset"]

    # ============================
    # 2) Fix seed
    # ============================
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ============================
    # 3) Resolver target (single-target para benchmark)
    # ============================
    target = dataset_cfg.get("target", None)
    targets_list = dataset_cfg.get("targets", None)

    if target is None:
        # Si solo te dieron targets=[...], fuerza que sea single o falla
        if isinstance(targets_list, list) and len(targets_list) == 1:
            target = targets_list[0]
        elif isinstance(targets_list, list) and len(targets_list) > 1:
            raise ValueError(
                "Para benchmark/entrenamiento single-target define dataset.target en el YAML "
                "(ej: target: u0). Tienes dataset.targets con múltiples valores."
            )
        else:
            target = "u0"

    print(f"[INFO] Target a predecir (single): {target}")

    # ============================
    # 4) Crear Dataset completo
    # ============================
    dataset = QM9SDFDataset(
        sdf_path=dataset_cfg["sdf"],
        csv_path=dataset_cfg["csv"],
        target_col=target,
    )

    # ============================
    # 5) Submuestreo (opcional)
    # ============================
    sample_size = cfg.get("sample_size", None)
    if sample_size is not None:
        random.seed(seed)
        total = len(dataset)
        sample_size = min(sample_size, total)
        indices = random.sample(range(total), sample_size)
        dataset = Subset(dataset, indices)
        print(f"[INFO] Usando {sample_size} muestras aleatorias de {total} disponibles.")

    # ============================
    # 6) Split train/val/test
    # ============================
    benchmark_split = bool(cfg.get("benchmark_split", True))
    n_total = len(dataset)
    train_idx, val_idx, test_idx = make_splits(n_total=n_total, seed=seed, benchmark_split=benchmark_split)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)
    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Guardar split para predict/evaluación reproducible
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
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[INFO] Split guardado en: {split_path}")

    # ============================
    # 7) DataLoaders
    # ============================
    num_workers = cfg.get("num_workers", 4)
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_mol,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_mol,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if max(1, num_workers // 2) > 0 else False,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_mol,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if max(1, num_workers // 2) > 0 else False,
    )

    # ============================
    # 8) Modelo
    # ============================
    model = ComplexPolarTransformerBeta(
        in_dim=cfg["model"]["in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        out_dim=cfg["model"].get("out_dim", 1),
        num_hidden_layers=cfg["model"].get("num_hidden_layers", 1),
        num_rbf=cfg["model"].get("num_rbf", 50),
        cutoff=float(cfg["model"].get("cutoff", 5.0)),
        edge_dim=cfg["model"].get("edge_dim", 4),
        dropout=cfg["model"].get("dropout", 0.0),
        use_residuals=cfg["model"].get("use_residuals", False),
        use_layernorm=cfg["model"].get("use_layernorm", False),
    )

    # ============================
    # 9) Entrenador
    # ============================
    early = cfg.get("early_stopping", {}) or {}
    sched = cfg.get("scheduler", None)

    hparams = {
        "model": cfg.get("model", {}),
        "learning_rate": cfg.get("learning_rate"),
        "batch_size": cfg.get("batch_size"),
        "max_epochs": cfg.get("max_epochs"),
        "seed": seed,
        "num_workers": num_workers,
        "target": target,
        "split_path": split_path,
        "benchmark_split": benchmark_split,
    }

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        lr=float(cfg["learning_rate"]),
        max_epochs=cfg["max_epochs"],
        ckpt_dir=cfg.get("ckpt_dir", "checkpoints"),
        log_dir=cfg.get("log_dir", "logs"),
        hparams=hparams,
        patience=int(early.get("patience", 30)),
        min_delta=float(early.get("min_delta", 5e-4)),
        scheduler_cfg=sched,
    )

    # ============================
    # 10) Ejecutar entrenamiento
    # ============================
    trainer.fit()