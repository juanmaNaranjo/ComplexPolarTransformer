import math
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

# ─────────────────────────────────────────────
# Propiedades atómicas para H, C, N, O, F
# Columnas: [masa(u), electronegatividad, valencia_max, radio_covalente(Å)]
# ─────────────────────────────────────────────
ATOM_PROPS = {
    1: [1.008,  2.20, 1, 0.31],   # H
    6: [12.011, 2.55, 4, 0.76],   # C
    7: [14.007, 3.04, 3, 0.71],   # N
    8: [15.999, 3.44, 2, 0.66],   # O
    9: [18.998, 3.98, 1, 0.57],   # F
}
# Rangos para normalización [0, 1]
_PROP_MIN = [1.008,  2.20, 1, 0.31]
_PROP_MAX = [18.998, 3.98, 4, 0.76]


class QM9SDFDataset(Dataset):
    """
    Dataset QM9 con representación geométrica polar centrada.

    Versión v8:
    - Features atómicas: one-hot 5-dim + 4 propiedades normalizadas = 9 dims.
      Propiedades: masa atómica, electronegatividad, valencia_max, radio covalente.
    - edge_attr[:, 0]: distancia real en Å para RBFExpansion.
    - edge_attr[:, 2]: sin(Δθ)  diferencia angular polar.
    - edge_attr[:, 3]: sin(Δφ)  diferencia angular azimutal.
    - Invariancia traslacional: coordenadas centradas en el centroide.
    - Si una molécula queda sin aristas: tensores vacíos shape-safe.
    """

    def __init__(self, sdf_path, csv_path, target_col="u0_atom", max_radius=7.0):
        self.max_radius = float(max_radius)
        self.atom_list = [1, 6, 7, 8, 9]   # H, C, N, O, F

        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        df = pd.read_csv(csv_path)

        self.mols = []
        valid_rows = []
        self.original_indices = []
        self.num_atoms = []

        for i, mol in enumerate(suppl):
            if mol is None or i >= len(df):
                continue
            self.mols.append(mol)
            valid_rows.append(df.iloc[i])
            self.original_indices.append(i)
            self.num_atoms.append(mol.GetNumAtoms())

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(f"Moléculas válidas: {len(self.mols)}, CSV sincronizado: {len(self.df)}")
        print(f"Radio máximo de aristas / cutoff dataset: {self.max_radius:.3f} Å")
        print("Columnas disponibles:", self.df.columns.tolist())

        if target_col in self.df.columns:
            self.target_col = target_col
        else:
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                raise ValueError("No hay columnas numéricas en el CSV.")
            self.target_col = numeric_cols[0]
            print(f"Usando '{self.target_col}' como target.")

    def __len__(self):
        return len(self.mols)

    def atom_to_one_hot(self, atomic_num: int) -> np.ndarray:
        """
        One-hot de tipo atómico (5 dims) + propiedades físico-químicas normalizadas (4 dims).
        Total: 9 dimensiones por átomo.
        """
        oh = np.zeros(5, dtype=np.float32)
        if atomic_num in self.atom_list:
            oh[self.atom_list.index(atomic_num)] = 1.0

        props = ATOM_PROPS.get(atomic_num, [1.0, 2.5, 1, 0.5])
        norm = [
            (props[i] - _PROP_MIN[i]) / (_PROP_MAX[i] - _PROP_MIN[i] + 1e-9)
            for i in range(4)
        ]
        return np.concatenate([oh, norm], dtype=np.float32)   # [9]

    @staticmethod
    def cart_to_spherical(xyz):
        x, y, z = xyz
        r = np.linalg.norm(xyz)
        theta = np.arccos(np.clip(z / (r + 1e-9), -1.0, 1.0))
        phi = np.arctan2(y, x)
        return r, theta, phi

    def __getitem__(self, idx):
        t0 = time.perf_counter()
        mol = self.mols[idx]
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        coords_cart = []
        atom_types = []

        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords_cart.append([pos.x, pos.y, pos.z])
            atom_types.append(self.atom_to_one_hot(mol.GetAtomWithIdx(i).GetAtomicNum()))

        coords_cart = np.asarray(coords_cart, dtype=np.float32)

        # Invariancia traslacional: centrar la molécula
        coords_cart -= coords_cart.mean(axis=0)
        coords_sph = np.asarray(
            [self.cart_to_spherical(xyz) for xyz in coords_cart],
            dtype=np.float32
        )

        coords_cart = torch.from_numpy(coords_cart).float()
        coords_sph  = torch.from_numpy(coords_sph).float()
        atom_types  = torch.from_numpy(np.asarray(atom_types, dtype=np.float32)).float()

        edge_index = []
        edge_attr  = []

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue

                diff = coords_cart[j] - coords_cart[i]
                dist = torch.norm(diff).item()

                if dist <= self.max_radius:
                    ri, ti, pi = coords_sph[i]
                    rj, tj, pj = coords_sph[j]

                    edge_index.append([i, j])
                    edge_attr.append([
                        dist,                                           # col 0: distancia real en Å
                        float((rj - ri) / self.max_radius),            # col 1: Δr normalizado
                        math.sin(float(tj - ti)),                      # col 2: sin(Δθ)
                        math.sin(float(pj - pi)),                      # col 3: sin(Δφ)
                    ])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, 4),  dtype=torch.float32)

        target = torch.tensor(float(self.df.iloc[idx][self.target_col]), dtype=torch.float32)
        sample_build_time_sec = time.perf_counter() - t0

        return {
            "coords_cart":        coords_cart,
            "coords_spherical":   coords_sph,
            "atom_types":         atom_types,
            "edge_index":         edge_index,
            "edge_attr":          edge_attr,
            "y":                  target,
            "sample_build_time_sec": sample_build_time_sec,
            "num_atoms":          num_atoms,
            "num_edges":          int(edge_index.shape[1]),
            "original_idx":       self.original_indices[idx],
        }
