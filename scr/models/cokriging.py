from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import LinearRegression


@dataclass
class CKConfig:
    
    lf_length: Tuple[float, float] = (20.0, 8.0)
    hf_length: Tuple[float, float] = (15.0, 6.0)
    lf_nugget: float = 1e-4
    hf_nugget: float = 1e-4
    nu: float = 1.5  # Matérn ν
    
    n_restarts: int = 2


def _make_matern_ard(length_xy: Tuple[float, float], nu: float, nugget: float):
    
    length_scale = np.asarray(length_xy, dtype=float)
    k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=length_scale, nu=nu)
    k += WhiteKernel(noise_level=nugget, noise_level_bounds=(1e-8, 1e-1))
    return k


class CoKrigingAR1:
    
    def __init__(self, cfg: Optional[CKConfig] = None):
        self.cfg = cfg or CKConfig()
        self.gp_L: Optional[GaussianProcessRegressor] = None
        self.gp_delta: Optional[GaussianProcessRegressor] = None
        self.rho_: Optional[float] = None

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X_L: np.ndarray, y_L: np.ndarray, X_H: np.ndarray, y_H: np.ndarray):
        
        X_L = self._ensure_2d(X_L)
        X_H = self._ensure_2d(X_H)
        y_L = np.asarray(y_L).ravel()
        y_H = np.asarray(y_H).ravel()

        kL = _make_matern_ard(self.cfg.lf_length, self.cfg.nu, self.cfg.lf_nugget)
        self.gp_L = GaussianProcessRegressor(kernel=kL, n_restarts_optimizer=self.cfg.n_restarts, normalize_y=True)
        self.gp_L.fit(X_L, y_L)

        mL_H = self.gp_L.predict(X_H, return_std=False)
        reg = LinearRegression().fit(mL_H.reshape(-1, 1), y_H)
        self.rho_ = float(reg.coef_[0])
        residual = y_H - self.rho_ * mL_H

        kD = _make_matern_ard(self.cfg.hf_length, self.cfg.nu, self.cfg.hf_nugget)
        self.gp_delta = GaussianProcessRegressor(kernel=kD, n_restarts_optimizer=self.cfg.n_restarts, normalize_y=True)
        self.gp_delta.fit(X_H, residual)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
      
        assert self.gp_L is not None and self.gp_delta is not None and self.rho_ is not None, "Model not fitted"
        X = self._ensure_2d(X)
        mL, sL = self.gp_L.predict(X, return_std=True)
        mD, sD = self.gp_delta.predict(X, return_std=True)
        mean = self.rho_ * mL + mD
        if return_std:
            std = np.sqrt((self.rho_ ** 2) * (sL ** 2) + (sD ** 2))  # 近似独立相加
            return mean, std
        return mean

