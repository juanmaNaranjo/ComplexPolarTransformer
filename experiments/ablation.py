"""
Estudio de ablación: mide el impacto individual de cada componente.
Ejecutar: python experiments/ablation.py
"""
import yaml, torch, random, json, os
from torch.utils.data import DataLoader, random_split, Subset
from core.dataset import QM9SDFDataset
from core.collate import pyg_collate
from core.trainer import Trainer
from core.utils import set_seed
from models.complex_model_beta import ComplexPolarTransformerBeta

CONFIGS = {
    "full_model":        {"use_residuals": True,  "use_layernorm": True,  "num_hidden_layers": 2},
    "no_layernorm":      {"use_residuals": True,  "use_layernorm": False, "num_hidden_layers": 2},
    "no_residuals":      {"use_residuals": False, "use_layernorm": True,  "num_hidden_layers": 2},
    "single_layer":      {"use_residuals": True,  "use_layernorm": True,  "num_hidden_layers": 1},
}

def run_ablation():
    cfg = yaml.safe_load(open("experiments/beta_train.yaml"))
    set_seed(42)

    dataset = QM9SDFDataset(
        sdf_path=cfg["dataset"]["sdf"],
        csv_path=cfg["dataset"]["csv"],
        target_col=cfg["dataset"].get("target", "u0"),
    )

    sample_size = cfg.get("sample_size", 5000)
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    dataset = Subset(dataset, indices)

    n = len(dataset)
    n_test  = int(n * 0.10)
    n_val   = int(n * 0.10)
    n_train = n - n_val - n_test
    train_ds, val_ds, _ = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    results = {}
    for name, variant in CONFIGS.items():
        print(f"\n{'='*40}\nAblación: {name}\n{'='*40}")
        model = ComplexPolarTransformerBeta(
            in_dim=cfg["model"]["in_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            out_dim=cfg["model"]["out_dim"],
            **variant,
        )
        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, collate_fn=pyg_collate)
        val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              collate_fn=pyg_collate)
        trainer = Trainer(
            model=model, train_dl=train_dl, val_dl=val_dl,
            lr=float(cfg["learning_rate"]),
            max_epochs=min(cfg["max_epochs"], 30),
            ckpt_dir=f"checkpoints/ablation_{name}",
            log_dir=f"logs/ablation_{name}",
        )
        trainer.fit()
        results[name] = trainer.best_val

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Resultados de ablación ===")
    for k, v in results.items():
        print(f"  {k}: best_val_mse = {v:.6f}")

if __name__ == "__main__":
    run_ablation()