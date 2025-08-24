import torch, math

def gaussian_nll(mean, var, y):
    return 0.5*(torch.log(2*math.pi*var) + (y-mean)**2/var)

def coverage(y, mean, std, k:float=2.0):
    lo, hi = mean - k*std, mean + k*std
    return ((y >= lo) & (y <= hi)).float().mean()

def crps_gaussian(mean, std, y):
    # 正态分布的 CRPS 闭式解
    z = (y - mean) / std.clamp_min(1e-9)
    phi = torch.exp(-0.5*z**2)/math.sqrt(2*math.pi)
    Phi = 0.5*(1 + torch.erf(z / math.sqrt(2.0)))
    return std * (2*phi + z*(2*Phi - 1) - 1/math.sqrt(math.pi))
