import torch
from torch_cluster import knn


def chunked_knn(coords, k=16, **_kwargs):
    """CUDA-accelerated KNN via torch_cluster.

    Args:
        coords: (N, 3) point coordinates (any device)
        k: number of neighbours (excluding self)

    Returns:
        indices: (N, k) neighbour indices into *coords*
    """
    # torch_cluster.knn returns (2, N*k): row=query, col=neighbor
    # k+1 because it includes self as nearest neighbour
    edge_index = knn(coords, coords, k=k + 1)
    col = edge_index[1].view(-1, k + 1)
    # drop self (first column)
    return col[:, 1:]


def faces_to_edge_index(faces, inverse, num_grid_points):
    """Convert mesh faces to an undirected edge_index at grid-sampled resolution.

    Args:
        faces: (F, 3) original-resolution face vertex indices
        inverse: (N_orig,) mapping original vertex → grid cell index
        num_grid_points: number of grid-sampled points

    Returns:
        edge_index: (2, E) undirected edges at grid resolution
    """
    grid_faces = inverse[faces]  # (F, 3)

    edges = torch.cat([
        grid_faces[:, [0, 1]],
        grid_faces[:, [1, 2]],
        grid_faces[:, [2, 0]],
    ], dim=0)  # (3F, 2)

    # remove self-loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]

    # make undirected
    edges = torch.cat([edges, edges.flip(1)], dim=0)

    # deduplicate
    edges = torch.unique(edges, dim=0)

    return edges.t().contiguous()  # (2, E)
