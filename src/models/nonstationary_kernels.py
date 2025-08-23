"""Nonstationary rotated RBF (Gibbs-like) / 非平稳旋转 RBF 核（到海岸距离控制相关长度）。"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
try:
    from shapely.geometry import Polygon, Point
except Exception:
    Polygon = None

class NonstationaryRotatedRBFMatrix:
    def __init__(self, vx: float, vy: float,
                 l_base: Tuple[float,float]=(30.0,8.0),
                 l_boost: Tuple[float,float]=(20.0,5.0),
                 boost_tau: float=8.0,
                 coast_polygon: Optional['Polygon']=None):
        self.vx, self.vy = float(vx), float(vy)
        self.R = self._rot(vx, vy)
        self.l_base = np.asarray(l_base, float)
        self.l_boost = np.asarray(l_boost, float)
        self.boost_tau = float(boost_tau)
        self.coast_poly = coast_polygon

    @staticmethod
    def _rot(vx, vy):
        t = float(np.arctan2(vy, vx)); c, s = np.cos(t), np.sin(t)
        return np.array([[c, s], [-s, c]], float)

    def _dist_to_coast(self, X: np.ndarray) -> np.ndarray:
        if self.coast_poly is None or Polygon is None:
            return np.full((X.shape[0],), 1e6, float)
        d = np.zeros((X.shape[0],), float)
        for i,(x,y) in enumerate(X):
            d[i] = self.coast_poly.exterior.distance(Point(float(x), float(y)))
        return d

    def _length_at(self, X: np.ndarray) -> np.ndarray:
        d = self._dist_to_coast(X)
        w = 1.0 - np.exp(-d / max(self.boost_tau, 1e-6))
        return self.l_base + self.l_boost * w[:,None]  # (N,2)

    def gram(self, X: np.ndarray, Y: Optional[np.ndarray]=None) -> np.ndarray:
        X = np.asarray(X, float); Xp = X @ self.R.T
        if Y is None: Y = X
        Y = np.asarray(Y, float); Yp = Y @ self.R.T
        lx = self._length_at(X); ly = self._length_at(Y)
        K = 1.0
        for k in range(2):
            Lsum = (lx[:,[k]]**2 + ly[:,k][None,:]**2)
            norm = np.sqrt(2.0 * lx[:,[k]] * ly[:,k][None,:] / (Lsum + 1e-12))
            diff = Xp[:,[k]] - Yp[:,k][None,:]
            K *= norm * np.exp(-(diff**2) / (Lsum + 1e-12))
        return K
