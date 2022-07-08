import torch
import torch.nn.functional as F

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()
    
    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

def mean_filter(img, r):
    device = img.device
    
    weight = torch.ones((2 * r + 1,), device=device)/(2 * r + 1)
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

def minconv(input):
    device = input.device
    disp_width = input.shape[-1]
    
    disp1d = torch.linspace(-(disp_width//2), disp_width//2, disp_width, device=device)
    regular1d = (disp1d.view(1,-1) - disp1d.view(-1,1)) ** 2
    
    output = torch.min( input.view(-1, disp_width, 1, disp_width, disp_width) + regular1d.view(1, disp_width, disp_width, 1, 1), 1)[0]
    output = torch.min(output.view(-1, disp_width, disp_width, 1, disp_width) + regular1d.view(1, 1, disp_width, disp_width, 1), 2)[0]
    output = torch.min(output.view(-1, disp_width, disp_width, disp_width, 1) + regular1d.view(1, 1, 1, disp_width, disp_width), 3)[0]
    
    output = output - (torch.min(output.view(-1, disp_width**3), 1)[0]).view(-1, 1, 1, 1)

    return output.view_as(input)