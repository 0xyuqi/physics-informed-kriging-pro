import torch
from typing import Optional
try:
    from shapely.geometry import LineString, Polygon
except Exception:
    LineString = Polygon = None

class BarrierAttenuation:
    def __init__(self, polygon: Optional[Polygon]=None, gamma:float=5.0, raster_mask: Optional[torch.Tensor]=None):
        self.polygon = polygon
        self.gamma = gamma
        self.mask = raster_mask  # [H,W] with 1 for land

    def _cross_poly(self, x1,y1,x2,y2)->bool:
        if self.polygon is None or LineString is None: return False
        line = LineString([(x1,y1),(x2,y2)])
        return line.crosses(self.polygon) or line.within(self.polygon) or (line.touches(self.polygon) and not line.disjoint(self.polygon))

    def _cross_mask(self, xs, ys):
        if self.mask is None: return False
        H,W = self.mask.shape
        ix = torch.clamp((xs*W).long(),0,W-1)
        iy = torch.clamp((ys*H).long(),0,H-1)
        return self.mask[iy,ix].any().item()

    def attenuation(self, X:torch.Tensor, X2:torch.Tensor)->torch.Tensor:
        # X: [N,3], X2: [M,3]; returns [N,M] multiplicative attenuation in (0,1]
        N,M = X.shape[0], X2.shape[0]
        att = torch.ones(N,M, device=X.device)
        if self.polygon is None and self.mask is None: return att
        K = 16
        t = torch.linspace(0,1,K, device=X.device)
        for i in range(N):
            for j in range(M):
                xs = X[i,0]*(1-t) + X2[j,0]*t
                ys = X[i,1]*(1-t) + X2[j,1]*t
                crossed = False
                if self.polygon is not None:
                    crossed = self._cross_poly(X[i,0].item(),X[i,1].item(),X2[j,0].item(),X2[j,1].item())
                if not crossed and self.mask is not None:
                    crossed = self._cross_mask(xs, ys)
                if crossed:
                    att[i,j] = torch.exp(torch.tensor(-self.gamma, device=X.device))
        return att
