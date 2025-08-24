import torch, torch.nn as nn

class PDEMean(nn.Module):
    """
    将离散的 [T,H,W] 场作为 GP 的先验均值：m(x,y,t) = α·u(x,y,t) + β
    通过 3D grid_sample 三线性插值采样。
    """
    def __init__(self, fields: torch.Tensor, xlim=(0.,1.), ylim=(0.,1.), tlim=(0.,1.)):
        super().__init__()
        assert fields.ndim == 3, "fields must be [T,H,W]"
        self.register_buffer("vol", fields[None,None])  # [1,1,T,H,W]
        self.register_buffer("xlim", torch.tensor(xlim, dtype=torch.float32))
        self.register_buffer("ylim", torch.tensor(ylim, dtype=torch.float32))
        self.register_buffer("tlim", torch.tensor(tlim, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N,3] -> [x,y,t]（物理坐标）
        x = (X[...,0]-self.xlim[0])/(self.xlim[1]-self.xlim[0])*2-1
        y = (X[...,1]-self.ylim[0])/(self.ylim[1]-self.ylim[0])*2-1
        t = (X[...,2]-self.tlim[0])/(self.tlim[1]-self.tlim[0])*2-1

        # 3D 体采样：vol 形状 [1,1,T,H,W]，grid 为 [1,1,N,1,3]
        g = torch.stack([x,y,t], dim=-1).view(1,1,-1,1,3)
        v = torch.nn.functional.grid_sample(self.vol, g, mode='bilinear', align_corners=True)
        vals = v.view(-1)
        return self.alpha*vals + self.beta
