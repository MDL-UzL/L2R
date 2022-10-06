from scipy.sparse import csr_matrix, csgraph
import torch
import torch.nn.functional as F
import time

from filters import *
from utils import *

def minimum_spanning_tree(dist):
    device = dist.device
    N = dist.shape[1]
    
    mst = csgraph.minimum_spanning_tree(csr_matrix(dist[0].cpu().numpy()))
    bfo = csgraph.breadth_first_order(mst, 0, directed=False)
    edges = torch.tensor([bfo[1][bfo[0]][1:], bfo[0][1:]], dtype=torch.long).t().view(1, -1, 2)
    
    level = torch.zeros((1, N, 1), dtype=torch.long)
    for i in range(N-1):
        level[0, edges[0, i, 1], 0] = level[0, edges[0, i, 0], 0] + 1
        
    idx = edges[0,:,1].sort()[1]
    edges = edges[:, idx, :]
        
    return edges.to(device), level.to(device)

def sym_knn_graph(dist, k):
    device = dist.device
    N = dist.shape[1]
    
    include_self=False
    ind = (-dist).topk(k + (1 - int(include_self)), dim=-1)[1][:, :, 1 - int(include_self):]
    A = torch.zeros((N, N), dtype=torch.bool, device=device)
    A[torch.arange(N).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[ind[0].t().contiguous().view(-1), torch.arange(N).repeat(k)] = 1
    
    edges = A.nonzero()
    edges_idx = torch.zeros_like(A, dtype=torch.long)
    edges_idx[A] = torch.arange(edges.shape[0], device=device)
    edges_reverse_idx = edges_idx.t()[A]
    
    return edges.unsqueeze(0), edges_reverse_idx.unsqueeze(0)
    
def kpts_dist(kpts, img, beta, k=64):
    device = kpts.device
    B, N, _ = kpts.shape
    _, _, D, H, W = img.shape
    k = min(k,N)
    
    dist = cdist(kpts_world(kpts, (D, H, W), align_corners=True), kpts_world(kpts, (D, H, W), align_corners=True), sqrt=True)
    dist[:, torch.arange(dist.shape[1], device=device), torch.arange(dist.shape[2], device=device)] = 1e15
    dist[dist<0.1] = 0.1
    img_mean = mean_filter(img, 2)
    kpts_mean = F.grid_sample(img_mean, kpts.view(1, 1, 1, -1, 3).to(img_mean.dtype), mode='nearest', align_corners=True).view(1, -1, 1)
    dist += cdist(kpts_mean, kpts_mean, p=1)/beta
    
    ind = (-dist).topk(k, dim=-1)[1]
    A = torch.zeros((B, N, N), device=device)
    A[:, torch.arange(N, device=device).repeat(k), ind[0].t().contiguous().view(-1)] = 1
    A[:, ind[0].t().contiguous().view(-1), torch.arange(N, device=device).repeat(k)] = 1
    dist = A*dist
    
    return dist

def random_kpts(mask, d, num_points=None):
    device = mask.device
    _, _, D, H, W = mask.shape
    
    kpts = torch.nonzero(mask[:, :, ::d, ::d, ::d]).unsqueeze(0).float()[:, :, 2:] * d
    
    if not num_points is None:
        kpts = kpts[:, torch.randperm(kpts.shape[1])[:num_points], :]
    
    return kpts_pt(kpts, (D, H, W), align_corners=True)