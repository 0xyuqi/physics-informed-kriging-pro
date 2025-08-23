"""合成数据生成器（含 proxy 低成本代理）/ Synthetic generator with low-cost proxy."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def flow_basis(vx, vy):
    v = np.array([vx,vy], float); ex = v/(np.linalg.norm(v)+1e-12); ey = np.array([-ex[1],ex[0]], float); return ex,ey
def gauss_elong(xy, center, ex, ey, Lp, Lt, amp=1.0):
    d = xy-center; s = d@ex; t = d@ey; return amp*np.exp(-(s/Lp)**2 - (t/Lt)**2)
def smooth2d(Z, k=2):
    Z=Z.copy()
    for _ in range(k):
        Z = 0.25*(np.pad(Z,(1,1),'edge')[1:-1,1:-1] + np.pad(Z,(1,1),'edge')[1:-1,0:-2] +                   np.pad(Z,(1,1),'edge')[0:-2,1:-1] + np.pad(Z,(1,1),'edge')[0:-2,0:-2])
    return Z

def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    L=a.grid; xs=np.linspace(0,100,L); ys=np.linspace(0,100,L); xx,yy=np.meshgrid(xs,ys,indexing='xy')
    XY=np.stack([xx,yy],-1).reshape(-1,2); ex,ey=flow_basis(a.vx,a.vy)
    s1,Lp1,Lt1,a1=np.array([30,40]),35,10,1.0; s2,Lp2,Lt2,a2=np.array([65,55]),25,8,0.6
    Z = gauss_elong(XY,s1,ex,ey,Lp1,Lt1,a1) + gauss_elong(XY,s2,ex,ey,Lp2,Lt2,a2)
    Z = (Z + a.bg_slope*(XY@ex)).reshape(L,L)
    rng=np.random.default_rng(a.seed); idx=rng.choice(L*L,size=a.n_obs,replace=False)
    ix,iy=idx%L,idx//L; x_obs=xs[ix]; y_obs=ys[iy]; z_obs=Z[iy,ix] + rng.normal(0.0,a.noise,size=a.n_obs)
    pd.DataFrame({'x':x_obs,'y':y_obs,'z':z_obs}).to_csv(out/'synth_points.csv',index=False)
    gi,gj=np.meshgrid(np.arange(L),np.arange(L),indexing='xy')
    pd.DataFrame({'i':gi.ravel(),'j':gj.ravel(),'x':xx.ravel(),'y':yy.ravel()}).to_csv(out/'grid_coords.csv',index=False)
    (out/'flow_meta.json').write_text(json.dumps({'vx':a.vx,'vy':a.vy,'grid':L,'noise':a.noise,'seed':a.seed,'bg_slope':a.bg_slope},indent=2))
    Zp = 0.7*Z + 0.3*smooth2d(Z,2) + 0.05*np.sin(xx/12.0) + rng.normal(0.0,0.03,size=Z.shape)
    pd.DataFrame({'x':xx.ravel(),'y':yy.ravel(),'proxy':Zp.ravel()}).to_csv(out/'proxy_grid.csv',index=False)
    sub=rng.choice(L*L,size=min(a.n_proxy,L*L),replace=False)
    pd.DataFrame({'x':xx.ravel()[sub],'y':yy.ravel()[sub],'proxy':Zp.ravel()[sub]}).to_csv(out/'proxy_points.csv',index=False)
    print(f"[OK] saved to {out}")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n_obs',type=int,default=40)
    ap.add_argument('--grid',type=int,default=80)
    ap.add_argument('--noise',type=float,default=0.1)
    ap.add_argument('--vx',type=float,default=1.0)
    ap.add_argument('--vy',type=float,default=0.3)
    ap.add_argument('--bg_slope',type=float,default=0.002)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--n_proxy',type=int,default=1200)
    ap.add_argument('--out_dir',type=str,default='data')
    main(ap.parse_args())
