import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset


class QM9SDFDataset(Dataset):
    """
    Dataset QM9 con representación geométrica polar centrada.

    v10 — correcciones de auditoría:
    - Neighbor search vectorizado con torch.cdist: elimina O(N²) Python loop
      con llamadas .item() por par.
    - edge_attr simplificado a [E, 1] (solo distancia en Å). RBFExpansion usa
      únicamente la columna 0; las 3 features angulares anteriores eran
      computadas pero descartadas, generando trabajo inútil.
    - Singularidad polar corregida: epsilon 1e-9 → 1e-6 (float32 ε_machine ≈ 1.2e-7;
      1e-9 estaba 3 órdenes por debajo de la precisión representable).
      Átomos con r < 1e-6 Å reciben theta=0, phi=0 explícitamente en lugar de
      producir valores indefinidos amplificados por división.
    - cart_to_spherical_batch: conversión vectorizada para N átomos a la vez.
    - Eliminado sample_build_time_sec: overhead de perf_counter en cada
      __getitem__ sin uso en el collate.
    - max_radius se recibe desde el YAML/modelo y define el radio real de aristas.
    - Si una molécula queda sin aristas se devuelven tensores vacíos seguros:
      edge_index=[2,0], edge_attr=[0,1].
    """

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

    def atom_to_one_hot(self, atomic_num):
        vec = np.zeros(len(self.atom_list), dtype=np.float32)
        if atomic_num in self.atom_list:
            vec[self.atom_list.index(atomic_num)] = 1.0
        return vec

    @staticmethod
    def cart_to_spherical_batch(xyz: np.ndarray) -> np.ndarray:
        """
        Conversión vectorizada cartesiana → esférica para [N, 3].

        Correcciones respecto a la versión escalar anterior:
        - Epsilon 1e-6 compatible con float32 (antes 1e-9 < ε_machine).
        - Singularidad explícita: r < 1e-6 → theta=0, phi=0 en lugar de
          producir valores arbitrarios o amplificar ruido numérico.
        - Vectorizada sobre N átomos en una sola llamada numpy.

        Returns: [N, 3] con columnas (r, theta, phi).
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.linalg.norm(xyz, axis=1)

        singular = r < 1e-6
        r_safe = np.where(singular, 1.0, r)

        theta = np.where(singular, 0.0, np.arccos(np.clip(z / r_safe, -1.0, 1.0)))
        phi = np.where(singular, 0.0, np.arctan2(y, x))

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
        atom_types = np.asarray(
            [self.atom_to_one_hot(mol.GetAtomWithIdx(i).GetAtomicNum())
             for i in range(num_atoms)],
            dtype=np.float32,
        )

        # Invariancia traslacional: centrar en el centroide geométrico.
        coords_cart -= coords_cart.mean(axis=0)

        # Coordenadas esféricas vectorizadas con corrección de singularidad.
        coords_sph = self.cart_to_spherical_batch(coords_cart)

        coords_cart_t = torch.from_numpy(coords_cart).float()
        coords_sph_t = torch.from_numpy(coords_sph).float()
        atom_types_t = torch.from_numpy(atom_types).float()

        # Neighbor search vectorizado con torch.cdist.
        # Elimina el O(N²) Python loop con .item() por par que era el cuello de
        # botella principal de CPU (~20M iteraciones Python por epoch completo).
        if num_atoms > 1:
            pos = torch.from_numpy(coords_cart)                               # [N, 3]
            dist_matrix = torch.cdist(
                pos.unsqueeze(0), pos.unsqueeze(0), p=2.0
            ).squeeze(0)                                                       # [N, N]

            mask = (dist_matrix > 0.0) & (dist_matrix <= self.max_radius)
            src_idx, dst_idx = mask.nonzero(as_tuple=True)                    # [E], [E]

            if src_idx.numel() > 0:
                edge_index = torch.stack([src_idx, dst_idx], dim=0).long()    # [2, E]
                # edge_attr: solo distancia real en Å (columna 0 usada por RBFExpansion).
                # Las 3 features angulares anteriores eran descartadas silenciosamente.
                edge_attr = dist_matrix[src_idx, dst_idx].unsqueeze(1).float()  # [E, 1]
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        target = torch.tensor(
            float(self.df.iloc[idx][self.target_col]), dtype=torch.float32
        )

        return {
            "coords_cart": coords_cart_t,
            "coords_spherical": coords_sph_t,
            "atom_types": atom_types_t,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": target,
            "num_atoms": num_atoms,
            "num_edges": int(edge_index.shape[1]),
            "original_idx": self.original_indices[idx],
        }
