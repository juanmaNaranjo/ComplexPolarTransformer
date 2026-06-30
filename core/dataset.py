import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch.utils.data import Dataset


class QM9SDFDataset(Dataset):
    """
    Dataset QM9 con representación geométrica polar centrada.

    v11 — atom features enriquecidas (auditoría científica):
    - atom_to_features: 5 → 12 dimensiones por átomo.
      [0:5]  one-hot número atómico (H, C, N, O, F)
      [5:8]  one-hot hibridación (SP, SP2, SP3)
      [8]    aromaticidad (bool)
      [9]    membresía en anillo (bool)
      [10]   carga formal normalizada (/ 2.0)
      [11]   número de hidrógenos totales normalizado (/ 4.0)
      La hibridación distingue isómeros estructurales; la aromaticidad
      es crítica para HOMO-LUMO gap; los anillos afectan la rigidez.

    Cambios heredados de v10:
    - Neighbor search vectorizado con torch.cdist (elimina O(N²) Python loop).
    - edge_attr = [E, 1] (distancia real en Å; RBFExpansion solo usa columna 0).
    - cart_to_spherical_batch: vectorizado, singularidad r<1e-6 → theta=phi=0.
    - max_radius desde YAML/modelo.
    """

    # Hibridación: SP=0, SP2=1, SP3=2 (one-hot en posiciones [5,6,7])
    _HYBRIDIZATION = {
        rdchem.HybridizationType.SP:  0,
        rdchem.HybridizationType.SP2: 1,
        rdchem.HybridizationType.SP3: 2,
    }

    def __init__(self, sdf_path, csv_path, target_col="u0", max_radius=5.0):
        self.max_radius = float(max_radius)
        self.atom_list = [1, 6, 7, 8, 9]  # H, C, N, O, F

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

    def atom_to_features(self, mol, atom_idx: int) -> np.ndarray:
        """
        Extrae 12 features físicas por átomo desde RDKit.

        Dimensiones:
          [0:5]  one-hot número atómico (H=1, C=6, N=7, O=8, F=9)
          [5:8]  one-hot hibridación (SP, SP2, SP3)
          [8]    is_aromatic     — crítico para gap HOMO-LUMO
          [9]    is_in_ring      — afecta rigidez y conjugación
          [10]   formal_charge / 2.0  — normalizado al rango típico [-1,1]
          [11]   total_num_Hs  / 4.0  — normalizado (máx H en QM9 ≈ 4)
        """
        atom = mol.GetAtomWithIdx(atom_idx)
        vec = np.zeros(12, dtype=np.float32)

        # One-hot número atómico
        an = atom.GetAtomicNum()
        if an in self.atom_list:
            vec[self.atom_list.index(an)] = 1.0

        # One-hot hibridación
        hyb_idx = self._HYBRIDIZATION.get(atom.GetHybridization(), -1)
        if hyb_idx >= 0:
            vec[5 + hyb_idx] = 1.0

        # Features binarias
        vec[8] = float(atom.GetIsAromatic())
        vec[9] = float(atom.IsInRing())

        # Features continuas normalizadas
        vec[10] = float(atom.GetFormalCharge()) / 2.0
        vec[11] = float(atom.GetTotalNumHs()) / 4.0

        return vec

    # Compatibilidad hacia atrás (no usada en __getitem__ nuevo)
    def atom_to_one_hot(self, atomic_num):
        vec = np.zeros(len(self.atom_list), dtype=np.float32)
        if atomic_num in self.atom_list:
            vec[self.atom_list.index(atomic_num)] = 1.0
        return vec

    @staticmethod
    def cart_to_spherical_batch(xyz: np.ndarray) -> np.ndarray:
        """
        Conversión vectorizada cartesiana → esférica para [N, 3].
        Epsilon 1e-6 compatible con float32; singularidad explícita r<1e-6 → θ=φ=0.
        Returns: [N, 3] con columnas (r, theta, phi).
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.linalg.norm(xyz, axis=1)

        singular = r < 1e-6
        r_safe = np.where(singular, 1.0, r)

        theta = np.where(singular, 0.0, np.arccos(np.clip(z / r_safe, -1.0, 1.0)))
        phi   = np.where(singular, 0.0, np.arctan2(y, x))

        return np.stack([r, theta, phi], axis=-1).astype(np.float32)

    def __getitem__(self, idx):
        mol = self.mols[idx]
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        coords_cart = np.asarray(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z]
             for i in range(num_atoms)],
            dtype=np.float32,
        )
        # v11: 12 features físicas por átomo (vs. 5 one-hot anteriores)
        atom_types = np.asarray(
            [self.atom_to_features(mol, i) for i in range(num_atoms)],
            dtype=np.float32,
        )  # [N, 12]

        # Invariancia traslacional: centrar en el centroide geométrico.
        coords_cart -= coords_cart.mean(axis=0)

        # Coordenadas esféricas vectorizadas con corrección de singularidad.
        coords_sph = self.cart_to_spherical_batch(coords_cart)

        coords_cart_t = torch.from_numpy(coords_cart).float()
        coords_sph_t  = torch.from_numpy(coords_sph).float()
        atom_types_t  = torch.from_numpy(atom_types).float()

        # Neighbor search vectorizado con torch.cdist — O(N²) en CUDA, no Python.
        if num_atoms > 1:
            pos = torch.from_numpy(coords_cart)                            # [N, 3]
            dist_matrix = torch.cdist(
                pos.unsqueeze(0), pos.unsqueeze(0), p=2.0
            ).squeeze(0)                                                   # [N, N]

            mask = (dist_matrix > 0.0) & (dist_matrix <= self.max_radius)
            src_idx, dst_idx = mask.nonzero(as_tuple=True)

            if src_idx.numel() > 0:
                edge_index = torch.stack([src_idx, dst_idx], dim=0).long()  # [2, E]
                edge_attr  = dist_matrix[src_idx, dst_idx].unsqueeze(1).float()  # [E, 1]
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr  = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, 1), dtype=torch.float32)

        target = torch.tensor(
            float(self.df.iloc[idx][self.target_col]), dtype=torch.float32
        )

        return {
            "coords_cart":    coords_cart_t,
            "coords_spherical": coords_sph_t,
            "atom_types":     atom_types_t,
            "edge_index":     edge_index,
            "edge_attr":      edge_attr,
            "y":              target,
            "num_atoms":      num_atoms,
            "num_edges":      int(edge_index.shape[1]),
            "original_idx":   self.original_indices[idx],
        }
