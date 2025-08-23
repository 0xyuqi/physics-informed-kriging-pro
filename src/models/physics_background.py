"""稳态对流-扩散 PDE 背景均值 / Steady advection–diffusion background mean."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np, pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
try:
    from shapely.geometry import Polygon, Point
except Exception:
    Polygon = None

@dataclass
class PDEConfig:
    vx: float; vy: float; kappa: float=2.0; c_in: float=0.0
    source_amp: float=0.0; source_xy: Optional[Tuple[float,float]]=None
    dx: float=1.0; dy: float=1.0

def _inflow_mask(vx, vy):
    m = {'left':False,'right':False,'bottom':False,'top':False}
    if vx>0: m['left']=True
    if vx<0: m['right']=True
    if vy>0: m['bottom']=True
    if vy<0: m['top']=True
    return m

def _build_source(nx, ny, xs, ys, cfg: PDEConfig):
    f = np.zeros((ny,nx), float)
    if cfg.source_amp!=0.0 and cfg.source_xy is not None:
        x0,y0 = cfg.source_xy
        sx = (xs - x0)[None,:]; sy = (ys - y0)[:,None]
        f = cfg.source_amp * np.exp(-(sx**2+sy**2)/(2.0*(3.0**2)))
    return f

def _mask_land(nx, ny, xx, yy, coast_poly):
    if coast_poly is None or Polygon is None:
        return np.zeros((ny,nx), bool)
    land = np.zeros((ny,nx), bool)
    for j in range(ny):
        for i in range(nx):
            if coast_poly.contains(Point(float(xx[j,i]), float(yy[j,i]))):
                land[j,i]=True
    return land

def solve_background(grid_df: pd.DataFrame, vx: float, vy: float,
                     kappa: float=2.0, c_in: float=0.0,
                     coast_polygon: Optional['Polygon']=None,
                     source_amp: float=0.0, source_xy: Optional[Tuple[float,float]]=None):
    nx = int(grid_df['i'].max()+1); ny = int(grid_df['j'].max()+1)
    xs = np.unique(grid_df['x'].values); ys = np.unique(grid_df['y'].values)
    xx = grid_df['x'].values.reshape(ny,nx); yy = grid_df['y'].values.reshape(ny,nx)
    dx = float(xs[1]-xs[0]) if len(xs)>1 else 1.0
    dy = float(ys[1]-ys[0]) if len(ys)>1 else 1.0
    cfg = PDEConfig(vx=vx, vy=vy, kappa=kappa, c_in=c_in, dx=dx, dy=dy,
                    source_amp=source_amp, source_xy=source_xy)
    land = _mask_land(nx, ny, xx, yy, coast_polygon)
    inflow = _inflow_mask(vx, vy)
    f = _build_source(nx, ny, xs, ys, cfg)

    N = nx*ny
    A = lil_matrix((N,N), float); b = np.zeros(N, float)
    def idx(i,j): return j*nx+i

    ax = max(vx,0.0)/dx; bx = -min(vx,0.0)/dx
    ay = max(vy,0.0)/dy; by = -min(vy,0.0)/dy
    cx = cfg.kappa/(dx*dx); cy = cfg.kappa/(dy*dy)

    for j in range(ny):
        for i in range(nx):
            p = idx(i,j)
            if land[j,i]: A[p,p]=1.0; b[p]=0.0; continue
            onL, onR = (i==0), (i==nx-1); onB, onT = (j==0), (j==ny-1)
            if (onL and inflow['left']) or (onR and inflow['right']) or (onB and inflow['bottom']) or (onT and inflow['top']):
                A[p,p]=1.0; b[p]=cfg.c_in; continue
            iL=max(i-1,0); iR=min(i+1,nx-1); jB=max(j-1,0); jT=min(j+1,ny-1)
            pL, pR, pB, pT = idx(iL,j), idx(iR,j), idx(i,jB), idx(i,jT)
            A[p,p]   = (ax+bx+ay+by) + 2*(cx+cy)
            A[p,pL] += -ax - cx
            A[p,pR] += -bx - cx
            A[p,pB] += -ay - cy
            A[p,pT] += -by - cy
            b[p] = f[j,i]
    c_vec = spsolve(csr_matrix(A), b)
    return c_vec.reshape(ny,nx)
