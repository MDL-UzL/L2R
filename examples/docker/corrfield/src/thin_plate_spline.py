import math
import torch
import torch.nn.functional as F

class TPS:       
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device
        
        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n+4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n+4, n+4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.linalg.solve(A, v)
        return theta
        
    @staticmethod
    def d(a, b):
        ra = (a**2).sum(dim=1).view(-1, 1)
        rb = (b**2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r**2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()
    
def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_step_size=2**12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D//step, H//step, W//step
    
    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1), align_corners=True).view(-1, 3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)
    
    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1 * H1 * W1
    n = math.ceil(N/unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = tps.z(x2[j1:j2], x1[0], theta)
        
    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)
    
    return y2