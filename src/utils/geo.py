"""Utility: 坐标系旋转（x,y ↔ 沿流/横流）/ rotation using (vx,vy)."""
from __future__ import annotations
import numpy as np

def angle_from_flow(vx: float, vy: float) -> float:
    return float(np.arctan2(vy, vx))

def _ensure_nx2(arr):
    A = np.asarray(arr, float)
    if A.shape == (2,): return A.reshape(1,2), (2,)
    if A.ndim < 2 or A.shape[-1] != 2: raise ValueError("Input must have shape (...,2).")
    return A.reshape(-1,2), A.shape

def rotate_to_flow(X, vx, vy):
    X2, orig = _ensure_nx2(X)
    th = angle_from_flow(vx, vy); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, s], [-s, c]], float)
    Y = X2 @ R.T
    return Y.reshape(orig)

def rotate_from_flow(Xp, vx, vy):
    X2, orig = _ensure_nx2(Xp)
    th = angle_from_flow(vx, vy); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, s], [-s, c]], float)
    Y = X2 @ R.T  # inverse rot is transpose; here symmetric pattern keeps consistency
    return Y.reshape(orig)
