import torch
from torch.utils.data import Dataset
from rdkit import Chem
import pandas as pd
import numpy as np
import math


class QM9SDFDataset(Dataset):
    """
    Dataset QM9 con representación geométrica polar centrada.
    """

    def __init__(self, sdf_path, csv_path, target_col="u0", max_radius=5.0):
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        df = pd.read_csv(csv_path)

        self.mols = []
        valid_rows = []

        for i, mol in enumerate(suppl):
            if mol is None or i >= len(df):
                continue
            self.mols.append(mol)
            valid_rows.append(df.iloc[i])

        self.df = pd.DataFrame(valid_rows)
        print(f"Moléculas válidas: {len(self.mols)}, CSV sincronizado: {len(self.df)}")
        print("Columnas disponibles:", self.df.columns.tolist())

        self.max_radius = max_radius
        self.atom_list = [1, 6, 7, 8, 9]  # H, C, N, O, F

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

    def atom_to_one_hot(self, atomic_num):
        vec = np.zeros(len(self.atom_list), dtype=np.float32)
        if atomic_num in self.atom_list:
            vec[self.atom_list.index(atomic_num)] = 1.0
        return vec

    def cart_to_spherical(self, xyz):
        x, y, z = xyz
        r = np.linalg.norm(xyz)
        theta = np.arccos(np.clip(z / (r + 1e-9), -1.0, 1.0))
        phi = np.arctan2(y, x)
        return r, theta, phi

    def __getitem__(self, idx):
        mol = self.mols[idx]
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        coords_cart = []
        atom_types = []

        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords_cart.append([pos.x, pos.y, pos.z])
            atom_types.append(self.atom_to_one_hot(mol.GetAtomWithIdx(i).GetAtomicNum()))

        coords_cart = np.array(coords_cart, dtype=np.float32)

        # 🔹 Centrar la molécula (invariancia traslacional)
        center = coords_cart.mean(axis=0)
        coords_cart -= center

        coords_sph = [self.cart_to_spherical(xyz) for xyz in coords_cart]

        coords_cart = torch.from_numpy(coords_cart).float()
        coords_sph = torch.from_numpy(np.asarray(coords_sph, dtype=np.float32))
        atom_types = torch.from_numpy(np.asarray(atom_types, dtype=np.float32))

        # 🔹 Construcción eficiente de edges
        edge_index = []
        edge_attr = []

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
                        dist / self.max_radius,            # normalizado
                        (rj - ri) / self.max_radius,
                        math.sin(tj - ti),
                        math.sin(pj - pi),
                    ])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        target = torch.tensor(float(self.df.iloc[idx][self.target_col]), dtype=torch.float32)

        return {
            "coords_cart": coords_cart,
            "coords_spherical": coords_sph,
            "atom_types": atom_types,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": target,
        }
