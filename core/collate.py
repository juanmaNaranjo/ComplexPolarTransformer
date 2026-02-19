import torch

def pyg_collate(batch):
    """
    Collate tipo PyTorch Geometric.
    Convierte una lista de moléculas en un único grafo grande.
    
    NOTA:
    - Usar solo cuando el modelo consuma edge_index / edge_attr.
    """

    coords_cart = []
    coords_spherical = []
    atom_types = []
    edge_index = []
    edge_attr = []
    y = []

    offset = 0

    for data in batch:
        n_atoms = data["coords_cart"].size(0)

        coords_cart.append(data["coords_cart"])
        coords_spherical.append(data["coords_spherical"])
        atom_types.append(data["atom_types"])
        edge_attr.append(data["edge_attr"])

        edge_index.append(data["edge_index"] + offset)

        y.append(data["y"])
        offset += n_atoms

    return {
        "coords_cart": torch.cat(coords_cart, dim=0),
        "coords_spherical": torch.cat(coords_spherical, dim=0),
        "atom_types": torch.cat(atom_types, dim=0),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_attr": torch.cat(edge_attr, dim=0),
        "y": torch.stack(y),
    }
