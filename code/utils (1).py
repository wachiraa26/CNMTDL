import torch



def neighbor0(N=14):
    src_list, dst_list = [], []
    offsets = [(-1,-1), (-1,0), (-1,1),
               ( 0,-1),         ( 0,1),
               ( 1,-1), ( 1,0), ( 1,1)]
    
    
    for i in range(N):
        for j in range(N):
            u = i*N+j
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    v = ni*N+nj
                    src_list.append(u)
                    dst_list.append(v)
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index

def neighbor0_4(N=14):
    """
    4-neighborhood (von Neumann) adjacency on an N x N grid (no diagonals).
    Returns undirected edge_index [2, E] with unique edges.
    """
    src_list, dst_list = [], []
    offsets = [(-1,0), (1,0), (0,-1), (0,1)]

    for i in range(N):
        for j in range(N):
            u = i*N + j
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < N:
                    v = ni*N + nj
                    src_list.append(u)
                    dst_list.append(v)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index
    
def neighbor1(N=14):
    src_list, dst_list = [], []
    H = N * (N - 1)
    def eh(i, j):
        return i * (N - 1) + j
    def ev(i, j):
        return H + i * N + j
        
    for i in range(N):
        for j in range(N):
            inc = []

            # incident horizontal edges at vertex (i,j)
            if j > 0:
                inc.append(eh(i, j - 1))  
            if j < N - 1:
                inc.append(eh(i, j))      

            # incident vertical edges at vertex (i,j)
            if i > 0:
                inc.append(ev(i - 1, j))  
            if i < N - 1:
                inc.append(ev(i, j))      

            # all pairs among incident edges (a != b)
            m = len(inc)
            for a in range(m):
                for b in range(m):
                    if a != b:
                        src_list.append(inc[a])
                        dst_list.append(inc[b])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index

def neighbor2(N=14):
    return neighbor0(N-1)


def neighbor0_3d(N: int) -> torch.Tensor:
    """
    3D 0-cell adjacency for an N x N x N grid using 26-neighborhood.
    Nodes are indexed by: idx(i,j,k) = (i*N + j)*N + k.

    Returns:
        edge_index: LongTensor of shape [2, E] (undirected, no self-loops, unique edges)
    """
    # 26 offsets in {-1,0,1}^3 \ {(0,0,0)}
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1)
               for dy in (-1, 0, 1)
               for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]

    def idx(i, j, k):
        return (i * N + j) * N + k

    src_list, dst_list = [], []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                u = idx(i, j, k)
                for dx, dy, dz in offsets:
                    ni, nj, nk = i + dx, j + dy, k + dz
                    if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                        v = idx(ni, nj, nk)
                        src_list.append(u)
                        dst_list.append(v)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Make undirected by adding reverse edges, then unique
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index

def neighbor0_3d_6(N: int) -> torch.Tensor:
    """
    3D 0-cell adjacency for an N x N x N grid using 6-neighborhood (von Neumann).
    """
    offsets = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

    def idx(i, j, k):
        return (i * N + j) * N + k

    src_list, dst_list = [], []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                u = idx(i, j, k)
                for dx, dy, dz in offsets:
                    ni, nj, nk = i + dx, j + dy, k + dz
                    if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                        v = idx(ni, nj, nk)
                        src_list.append(u)
                        dst_list.append(v)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index
