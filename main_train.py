import os
import yaml
from core.collate import pyg_collate
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
        "y": torch.stack([item["y"] for item in batch])
    }


if __name__ == "__main__":

    # ============================
    # 1) Cargar YAML
    # ============================
    cfg_path = "experiments/beta_train.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path))
    dataset_cfg = cfg["dataset"]

    # ============================
    # 2) Crear Dataset completo
    # ============================
    dataset = QM9SDFDataset(
        sdf_path=dataset_cfg["sdf"],
        csv_path=dataset_cfg["csv"],
        target_col=dataset_cfg.get("target", "u0")
    )

    # ============================
    # 3) Submuestreo (opcional)
    # ============================
    sample_size = cfg.get("sample_size", None)

    if sample_size is not None:
        random.seed(42)
        total = len(dataset)
        sample_size = min(sample_size, total)

        indices = random.sample(range(total), sample_size)
        dataset = Subset(dataset, indices)

        print(f"[INFO] Usando {sample_size} muestras aleatorias de {total} disponibles.")

    # ============================
    # 4) Fix seed y preparación
    # ============================
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ============================
    # 5) Split train / val / test  (80 / 10 / 10)
    # ============================
    n_total = len(dataset)
    n_test  = int(n_total * 0.10)
    n_val   = int(n_total * 0.10)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"[INFO] Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # Guardar índices del test para reproducibilidad
    import json, os
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/test_indices.json", "w") as f:
        json.dump(test_ds.indices, f)
    print("[INFO] Índices de test guardados en checkpoints/test_indices.json")

    # ============================
    # 6) DataLoaders
    # ============================
    num_workers = cfg.get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=pyg_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        collate_fn=collate_mol,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
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
        hparams=hparams,
    )

    # ============================
    # 9) Ejecutar entrenamiento
    # ============================
    trainer.fit()




