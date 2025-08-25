import torch, gpytorch
from .barrier import BarrierAttenuation

class SeparableSTKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False
    def __init__(self, space_kernel, time_kernel, barrier: BarrierAttenuation|None=None, **kwargs):
        super().__init__(**kwargs)
        self.space_kernel = space_kernel
        self.time_kernel  = time_kernel
        self.barrier = barrier
    def forward(self, X, X2=None, diag=False, **params):
        X2 = X if X2 is None else X2
        xs, ts  = X[...,:-1], X[...,-1:]
        xs2,ts2 = X2[...,:-1],X2[...,-1:]
        Ks = self.space_kernel(xs, xs2, diag=diag)
        Kt = self.time_kernel(ts, ts2, diag=diag)
        K  = Ks * Kt
        if diag or self.barrier is None: return K
        att = self.barrier.attenuation(X, X2)
        return K * att
