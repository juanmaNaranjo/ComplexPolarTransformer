import torch


def mol_collate(batch):
    """Collate estándar para ComplexPolarTransformer con moléculas de tamaño variable."""
    return {
        "coords_cart": [item["coords_cart"] for item in batch],
        "coords_spherical": [item["coords_spherical"] for item in batch],
        "atom_types": [item["atom_types"] for item in batch],
        "edge_index": [item["edge_index"] for item in batch],
        "edge_attr": [item["edge_attr"] for item in batch],
        "y": torch.stack([item["y"] for item in batch]),
        "num_atoms": torch.tensor([item.get("num_atoms", item["atom_types"].shape[0]) for item in batch], dtype=torch.float32),
        "num_edges": torch.tensor([item.get("num_edges", item["edge_index"].shape[1]) for item in batch], dtype=torch.long),
        "original_idx": torch.tensor([item.get("original_idx", -1) for item in batch], dtype=torch.long),
    }


def pyg_collate(batch):
    """
    Collate tipo PyTorch Geometric.
    Convierte una lista de moléculas en un único grafo grande.
    Úsalo solo si el modelo espera un grafo batch global, no listas por molécula.
    """
    coords_cart = []
    coords_spherical = []
    atom_types = []
    edge_index = []
    edge_attr = []
    y = []
    batch_index = []

    offset = 0
    for mol_idx, data in enumerate(batch):
        n_atoms = data["coords_cart"].size(0)
        coords_cart.append(data["coords_cart"])
        coords_spherical.append(data["coords_spherical"])
        atom_types.append(data["atom_types"])
        y.append(data["y"])
        batch_index.append(torch.full((n_atoms,), mol_idx, dtype=torch.long))

        if data["edge_index"].numel() > 0:
            edge_index.append(data["edge_index"] + offset)
            edge_attr.append(data["edge_attr"])
        offset += n_atoms

    return {
        "coords_cart": torch.cat(coords_cart, dim=0),
        "coords_spherical": torch.cat(coords_spherical, dim=0),
        "atom_types": torch.cat(atom_types, dim=0),
        "edge_index": torch.cat(edge_index, dim=1) if edge_index else torch.empty((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(edge_attr, dim=0) if edge_attr else torch.empty((0, 1), dtype=torch.float32),
        "batch": torch.cat(batch_index, dim=0),
        "y": torch.stack(y),
    }
