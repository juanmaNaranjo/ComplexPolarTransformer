import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split, Subset
from models.complex_model_beta import ComplexPolarTransformerBeta
from core.dataset import QM9SDFDataset
from core.trainer import Trainer
from core.utils import set_seed
import random


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


if __name__ == "__main__":

    # ============================
    # 1) Cargar YAML
    # ============================
    cfg_path = "experiments/beta_train.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg["dataset"]

    # ============================
    # 2) Fix seed y preparación
    # ============================
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ============================
    # 3) Crear Dataset completo
    # ============================
    dataset = QM9SDFDataset(
        sdf_path=dataset_cfg["sdf"],
        csv_path=dataset_cfg["csv"],
        target_col=dataset_cfg.get("target", "u0"),
    )

    # ============================
    # 4) Submuestreo (opcional)
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
    # 5) Split train/val
    # ============================
    n_total = len(dataset)
    n_train = int(n_total * (1 - cfg["validation_split"]))
    n_val = n_total - n_train

    split_generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_generator)
    print(f"[INFO] Train: {n_train} | Val: {n_val}")

    # ============================
    # 6) DataLoaders
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
        collate_fn=collate_mol,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if max(1, num_workers // 2) > 0 else False,
    )

    # ============================
    # 7) Modelo
    # ============================
    model = ComplexPolarTransformerBeta(
        in_dim=cfg["model"]["in_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        out_dim=cfg["model"]["out_dim"],
        num_hidden_layers=cfg["model"].get("num_hidden_layers", 1),
        dropout=cfg["model"].get("dropout", 0.0),
        use_residuals=cfg["model"].get("use_residuals", False),
        use_layernorm=cfg["model"].get("use_layernorm", False),
    )

    # ============================
    # 8) Entrenador
    # ============================
    hparams = {
        "model": cfg.get("model", {}),
        "learning_rate": cfg.get("learning_rate"),
        "batch_size": cfg.get("batch_size"),
        "max_epochs": cfg.get("max_epochs"),
        "seed": seed,
        "num_workers": num_workers,
    }

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        lr=float(cfg["learning_rate"]),
        max_epochs=cfg["max_epochs"],
        ckpt_dir="checkpoints",
        log_dir="logs",
        hparams=hparams,
    )

    # ============================
    # 9) Ejecutar entrenamiento
    # ============================
    trainer.fit()
