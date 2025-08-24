import torch, gpytorch

class SeparableSTKernel(gpytorch.kernels.Kernel):
    """
    K((x,t),(x',t')) = Ks(x,x') * Kt(t,t')，可选 barrier_raster（1=屏障/陆地）进行穿越衰减。
    """
    has_lengthscale = False
    def __init__(self, space_kernel, time_kernel, barrier_raster: torch.Tensor|None=None, barrier_strength: float=5.0, **kwargs):
        super().__init__(**kwargs)
        self.space_kernel = space_kernel
        self.time_kernel  = time_kernel
        self.register_buffer("barrier", barrier_raster if barrier_raster is not None else None)
        self.barrier_strength = barrier_strength

    def _attenuation(self, X, X2):
        if self.barrier is None: return None
        K = 16
        b = self.barrier
        Hb, Wb = b.shape
        x1,y1 = X[...,0], X[...,1]
        x2,y2 = X2[...,0], X2[...,1]
        def sample_mask(a1,a2, n=K):
            t = torch.linspace(0,1,n, device=a1.device)
            return (a1[...,None]*(1-t) + a2[...,None]*t)
        xs = sample_mask(x1[:,None], x2[None,:])  # [N,M,K]
        ys = sample_mask(y1[:,None], y2[None,:])
        ix = torch.clamp((xs*Wb).long(), 0, Wb-1)
        iy = torch.clamp((ys*Hb).long(), 0, Hb-1)
        crossed = b[iy, ix].any(dim=-1)  # [N,M]
        atten = torch.where(crossed, torch.exp(torch.tensor(-self.barrier_strength, device=xs.device)),
                            torch.tensor(1.0, device=xs.device))
        return atten

    def forward(self, X, X2=None, diag=False, **params):
        X2 = X if X2 is None else X2
        xs, ts  = X[...,:-1], X[...,-1:]
        xs2,ts2 = X2[...,:-1],X2[...,-1:]
        Ks = self.space_kernel(xs, xs2, diag=diag)
        Kt = self.time_kernel(ts, ts2, diag=diag)
        K  = Ks * Kt
        if diag: return K
        att = self._attenuation(X,X2)
        if att is not None: K = K * att
        return K
