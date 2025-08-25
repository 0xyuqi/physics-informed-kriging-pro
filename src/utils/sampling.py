from __future__ import annotations
import torch

@torch.no_grad()
def greedy_epv_with_min_dist(Xcand:torch.Tensor, var:torch.Tensor, k:int, dmin:float)->torch.Tensor:
    # Xcand: [M,2] or [M,3], var: [M]
    picked = []
    m = Xcand.shape[0]
    mask = torch.ones(m, dtype=torch.bool, device=var.device)
    for _ in range(k):
        idx = torch.argmax(torch.where(mask, var, torch.tensor(-1e9, device=var.device))).item()
        if var[idx].item() <= 0:
            break
        picked.append(idx)
        # enforce min distance on first 2 dims (x,y)
        dx = Xcand[:,0] - Xcand[idx,0]
        dy = Xcand[:,1] - Xcand[idx,1]
        mask = mask & ((dx*dx + dy*dy) >= dmin*dmin)
    return torch.tensor(picked, dtype=torch.long, device=var.device)
