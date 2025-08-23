"""Advanced GP（终极版 / bilingual 注释）"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, WhiteKernel, ConstantKernel as C
try:
    from shapely.geometry import Polygon, LineString
except Exception:
    Polygon = None

from .nonstationary_kernels import NonstationaryRotatedRBFMatrix

def _rot(vx: float, vy: float):
    th = float(np.arctan2(vy, vx)); c, s = np.cos(th), np.sin(th)
    return np.array([[c, s], [-s, c]], float)

class RotatedRBF(Kernel):
    def __init__(self, vx: float, vy: float, length_scale=(30.0,8.0), length_scale_bounds=(1e-2,1e3)):
        self.vx, self.vy = float(vx), float(vy)
        self.length_scale = np.asarray(length_scale, float)
        self.length_scale_bounds = length_scale_bounds
        self.R = _rot(vx, vy)
    @property
    def hyperparameter_length_scale(self): return Hyperparameter("length_scale","numeric",self.length_scale_bounds,len(self.length_scale))
    def diag(self, X): return np.ones(X.shape[0])
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, float); Xp = X @ self.R.T
        Yp = Xp if Y is None else np.asarray(Y, float) @ self.R.T
        ls = np.maximum(self.length_scale, 1e-12)
        Xn, Yn = Xp/ls, Yp/ls
        d2 = np.sum(Xn**2,1)[:,None] + np.sum(Yn**2,1)[None,:] - 2.0*(Xn @ Yn.T)
        K = np.exp(-0.5*d2)
        return (K, np.zeros((K.shape[0],K.shape[1],len(ls)))) if eval_gradient else K
    def is_stationary(self): return True

class RotatedRQ(Kernel):
    def __init__(self, vx: float, vy: float, length_scale=(30.0,8.0), alpha=0.5,
                 length_scale_bounds=(1e-2,1e3), alpha_bounds=(1e-3,1e3)):
        self.vx, self.vy = float(vx), float(vy)
        self.length_scale = np.asarray(length_scale, float)
        self.alpha = float(alpha); self.length_scale_bounds = length_scale_bounds; self.alpha_bounds = alpha_bounds
        self.R = _rot(vx, vy)
    @property
    def hyperparameter_alpha(self): return Hyperparameter("alpha","numeric",self.alpha_bounds)
    def diag(self, X): return np.ones(X.shape[0])
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, float); Xp = X @ self.R.T
        Yp = Xp if Y is None else np.asarray(Y, float) @ self.R.T
        ls = np.maximum(self.length_scale, 1e-12)
        Xn, Yn = Xp/ls, Yp/ls
        d2 = np.sum(Xn**2,1)[:,None] + np.sum(Yn**2,1)[None,:] - 2.0*(Xn @ Yn.T)
        base = 1.0 + 0.5*d2/max(self.alpha,1e-12)
        K = base ** (-self.alpha)
        return (K, np.zeros((K.shape[0],K.shape[1],1))) if eval_gradient else K
    def is_stationary(self): return True

class BarrierKernel(Kernel):
    """沿线段检查是否跨越海岸线；若跨越，相关性乘以 exp(-gamma)。/ Segment crosses coastline -> decay."""
    def __init__(self, coast: Optional['Polygon']=None, gamma: float=5.0):
        self.coast = coast; self.gamma = float(gamma)
    def set_coast(self, coast): self.coast = coast
    def diag(self, X): return np.ones(X.shape[0])
    def __call__(self, X, Y=None, eval_gradient=False):
        if self.coast is None or Polygon is None:
            M = np.ones((X.shape[0], X.shape[0] if Y is None else Y.shape[0]))
            return (M, np.zeros((*M.shape,0))) if eval_gradient else M
        X = np.asarray(X, float); Y = X if Y is None else np.asarray(Y, float)
        M = np.ones((X.shape[0], Y.shape[0]))
        for i, xi in enumerate(X):
            for j, yj in enumerate(Y):
                seg = LineString([tuple(xi), tuple(yj)])
                if seg.crosses(self.coast) or seg.touches(self.coast):
                    M[i,j] = np.exp(-self.gamma)
        return (M, np.zeros((*M.shape,0))) if eval_gradient else M
    def is_stationary(self): return False

class NonstationaryWrapper(Kernel):
    def __init__(self, builder: NonstationaryRotatedRBFMatrix): self.b = builder
    def diag(self, X): return np.ones(X.shape[0])
    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.b.gram(X, Y)
        return (K, np.zeros((*K.shape,0))) if eval_gradient else K
    def is_stationary(self): return False

def bilinear_interp(grid_xy: np.ndarray, values: np.ndarray, X: np.ndarray) -> np.ndarray:
    ny, nx, _ = grid_xy.shape
    xs = grid_xy[0,:,0]; ys = grid_xy[:,0,1]
    dx = xs[1]-xs[0] if nx>1 else 1.0; dy = ys[1]-ys[0] if ny>1 else 1.0
    x0, y0 = xs[0], ys[0]
    out = np.zeros((X.shape[0],), float)
    for n,(x,y) in enumerate(X):
        fx = (x-x0)/dx; fy = (y-y0)/dy
        i = int(np.clip(np.floor(fx), 0, nx-2)); j = int(np.clip(np.floor(fy), 0, ny-2))
        tx = fx - i; ty = fy - j
        v00 = values[j,i]; v10 = values[j,i+1]; v01 = values[j+1,i]; v11 = values[j+1,i+1]
        out[n] = (1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 + (1-tx)*ty*v01 + tx*ty*v11
    return out

@dataclass
class MeanProvider:
    mode: str = "pde"; vx: float = 1.0; vy: float = 0.0; use_quadratic: bool = True
    grid_xy: Optional[np.ndarray] = None; pde_field: Optional[np.ndarray] = None
    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.mode == "poly":
            R = _rot(self.vx, self.vy); ST = X @ R.T; s = ST[:,0:1]; t = ST[:,1:2]
            Phi = [np.ones_like(s), s, t]; 
            if self.use_quadratic: Phi += [s**2, t**2, s*t]
            Phi = np.concatenate(Phi, axis=1)
            beta, *_ = np.linalg.lstsq(Phi, y.reshape(-1,1), rcond=None)
            self.beta_ = beta.reshape(-1); self.R_ = R
        else:
            assert self.grid_xy is not None and self.pde_field is not None
            m = bilinear_interp(self.grid_xy, self.pde_field, X).reshape(-1,1)
            Phi = np.hstack([m, np.ones_like(m)])
            ab, *_ = np.linalg.lstsq(Phi, y.reshape(-1,1), rcond=None)
            self.ab_ = ab.reshape(-1)
    def mean(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "poly":
            ST = X @ self.R_.T; s = ST[:,0:1]; t = ST[:,1:2]
            Phi = [np.ones_like(s), s, t]; 
            if self.use_quadratic: Phi += [s**2, t**2, s*t]
            Phi = np.concatenate(Phi, axis=1); return (Phi @ self.beta_).ravel()
        else:
            m = bilinear_interp(self.grid_xy, self.pde_field, X).reshape(-1,1)
            a,b = self.ab_; return (a*m + b).ravel()

class AdvancedAnisotropicGP:
    def __init__(self, vx: float, vy: float,
                 lp: float=30.0, lc: float=8.0, rq_alpha: float=0.5,
                 noise_level: float=1e-3, learn_noise: bool=False,
                 mean_mode: str="pde", use_quadratic_trend: bool=True,
                 grid_xy: Optional[np.ndarray]=None, pde_field: Optional[np.ndarray]=None,
                 coast_polygon: Optional['Polygon']=None, barrier_gamma: float=5.0,
                 nonstat_boost: Tuple[float,float]=(20.0,5.0), nonstat_tau: float=8.0,
                 alpha_jitter: float=1e-6, optimize: bool=True, n_restarts: int=2, random_state: int=0):
        self.vx, self.vy = float(vx), float(vy)
        self.mean = MeanProvider(mean_mode, vx, vy, use_quadratic_trend, grid_xy, pde_field)
        k_rbf = RotatedRBF(vx, vy, (lp,lc))
        k_rq  = RotatedRQ(vx, vy, (lp,lc), alpha=rq_alpha)
        nonstat = NonstationaryWrapper(NonstationaryRotatedRBFMatrix(
            vx, vy, l_base=(lp,lc), l_boost=nonstat_boost, boost_tau=nonstat_tau, coast_polygon=coast_polygon
        ))
        barrier = BarrierKernel(coast_polygon, gamma=barrier_gamma)
        class _Sum(Kernel):
            def __init__(self,a:Kernel,b:Kernel): self.a=a; self.b=b
            def __call__(self,X,Y=None,eval_gradient=False): return self.a(X,Y,False)+self.b(X,Y,False)
            def diag(self,X): return self.a.diag(X)+self.b.diag(X)
            def is_stationary(self): return False
        kernel = C(1.0,(1e-3,1e3))*_Sum(k_rbf,k_rq)*barrier + nonstat + WhiteKernel(
            noise_level=noise_level, noise_level_bounds=(1e-6,1e-1) if learn_noise else "fixed"
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha_jitter, normalize_y=False,
            optimizer=('fmin_l_bfgs_b' if optimize else None),
            n_restarts_optimizer=(int(n_restarts) if optimize else 0),
            random_state=random_state
        )
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mean.fit(X, y); resid = y - self.mean.mean(X); self.gpr.fit(X, resid)
    def predict(self, X: np.ndarray, return_std: bool=True):
        X = np.asarray(X, float); mu_r, std_r = self.gpr.predict(X, return_std=True)
        mu = self.mean.mean(X) + mu_r; return (mu, std_r) if return_std else (mu, None)
