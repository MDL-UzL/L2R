import torch
import time

from filters import *

def tbp(cost, edges, level, dist):
    marginals = cost
    message = torch.zeros_like(marginals)
    
    for i in range(level.max(),0,-1):
        child = edges[0, level[0, 1:, 0]==i, 1]
        parent = edges[0, level[0, 1:, 0]==i, 0]
        weight = dist[0, child, parent].view(-1, 1, 1, 1)

        data = marginals[:, child, :, :, :]
        data_reg = minconv(data * weight) / weight

        message[:, child, :, :, :] = data_reg
        marginals.index_add_(1, parent, data_reg)

    for i in range(1, level.max()+1):
        child = edges[0, level[0, 1:, 0]==i, 1]
        parent = edges[0, level[0, 1:, 0]==i, 0]
        weight = dist[0, child, parent].view(-1, 1, 1, 1)

        data = marginals[:, parent, :, :, :] - message[:, child, :, :, :] + message[:, parent, :, :, :]
        data_reg = minconv(data * weight) / weight

        message[:, child, :, :, :] = data_reg

    marginals += message
    
    return marginals