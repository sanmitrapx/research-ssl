import torch
import pointops_cuda


def chunked_knn(coords, k=16, ref=None, **_kwargs):
    """CUDA-accelerated KNN via pointops.

    Args:
        coords: (N, 3) query point coordinates (CUDA)
        k: number of neighbours (excluding self for self-KNN)
        ref: (M, 3) optional reference points. If None, self-KNN is performed.

    Returns:
        indices: (N, k) LongTensor of neighbour indices into *ref* (or *coords*)
    """
    query = coords.float().contiguous()
    N = query.shape[0]

    if ref is None:
        xyz = query
        M = N
        nsample = k + 1
        drop_self = True
    else:
        xyz = ref.float().contiguous()
        M = xyz.shape[0]
        nsample = k
        drop_self = False

    offset_ref = torch.tensor([M], dtype=torch.int32, device=coords.device)
    offset_query = torch.tensor([N], dtype=torch.int32, device=coords.device)

    idx = torch.zeros(N, nsample, dtype=torch.int32, device=coords.device)
    dist2 = torch.zeros(N, nsample, dtype=torch.float32, device=coords.device)
    pointops_cuda.knnquery_cuda(N, nsample, xyz, query, offset_ref, offset_query, idx, dist2)

    if drop_self:
        return idx[:, 1:].long()
    return idx.long()


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
