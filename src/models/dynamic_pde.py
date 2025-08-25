import torch, math
from typing import List, Tuple

def _fft_freqs(H:int, W:int, device):
    kx = torch.fft.rfftfreq(W, d=1.0/W).to(device) * 2*math.pi
    ky = torch.fft.fftfreq(H,  d=1.0/H).to(device) * 2*math.pi
    return ky[:,None], kx[None,:]

@torch.no_grad()
def advect_diffuse_spectral(u0: torch.Tensor, v: Tuple[float,float], kappa: float, dt: float, steps: int) -> List[torch.Tensor]:
    H, W = u0.shape
    device = u0.device
    ky, kx = _fft_freqs(H,W,device)
    K2 = ky**2 + kx**2
    phase = torch.exp(-1j*(v[0]*kx + v[1]*ky)*dt)
    decay = torch.exp(-kappa*K2*dt)
    U = torch.fft.rfft2(u0)
    out = [u0.clone()]
    for _ in range(steps):
        U = U * phase * decay
        u = torch.fft.irfft2(U, s=(H,W)).real
        out.append(u)
    return out

def simulate_plume(H=128, W=128, T=25, v=(0.2,0.05), kappa=0.002, dt=0.2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    v = torch.tensor(v, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(torch.linspace(0,1,H,device=device),
                            torch.linspace(0,1,W,device=device), indexing='ij')
    cx, cy, sigma = 0.2, 0.6, 0.07
    u0 = torch.exp(-0.5*((xx-cx)**2 + (yy-cy)**2)/sigma**2)
    fields = advect_diffuse_spectral(u0, v=v, kappa=kappa, dt=dt, steps=T-1)
    return torch.stack(fields, dim=0)  # [T,H,W]
