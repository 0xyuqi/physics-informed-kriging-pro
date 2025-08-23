"""互信息近似 + 最小距离约束的主动采样 / MI-greedy active sampling."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def rbf_kernel(X, Y, ls=12.0):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    d2=((X[:,None,:]-Y[None,:,:])**2).sum(-1)
    return np.exp(-0.5*d2/(ls**2+1e-12))

def pick_next_points_mi(cand_xy, cand_std, taken_xy, k, min_dist, ls_corr):
    order=np.argsort(cand_std)[::-1]; chosen=[]; K=rbf_kernel(cand_xy,cand_xy,ls_corr); L=None; eps=1e-6
    for idx in order:
        p=cand_xy[idx]
        if taken_xy.shape[0]>0 and np.min(np.linalg.norm(taken_xy-p,axis=1))<min_dist: continue
        if any(np.linalg.norm(cand_xy[j]-p)<min_dist for j in chosen): continue
        if L is None: pass
        else:
            kvec=np.array([K[idx,j] for j in chosen],float)
            y=np.linalg.solve(L,kvec); _=np.log(max(K[idx,idx]-y.T@y,eps))
        chosen.append(idx)
        if L is None: L=np.array([[np.sqrt(K[idx,idx]+eps)]],float)
        else:
            kvec=np.array([K[idx,j] for j in chosen[:-1]],float); y=np.linalg.solve(L,kvec)
            diag=np.sqrt(max(K[idx,idx]+eps - y.T@y, eps))
            L=np.block([[L, np.zeros((L.shape[0],1))],[y[None,:], diag.reshape(1,1)]])
        if len(chosen)>=k: break
    return np.array(chosen,int)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',type=str,default='data'); ap.add_argument('--out_dir',type=str,default='figures')
    ap.add_argument('--k_next',type=int,default=8); ap.add_argument('--min_dist',type=float,default=3.0); ap.add_argument('--ls_corr',type=float,default=12.0)
    a=ap.parse_args()
    data_dir, out_dir = Path(a.data_dir), Path(a.out_dir)
    pred=pd.read_csv(data_dir/'grid_pred.csv'); pts=pd.read_csv(data_dir/'synth_points.csv')
    Xg=pred[['x','y']].to_numpy(); std=pred['std_y'].to_numpy(); Xt=pts[['x','y']].to_numpy()
    idx=pick_next_points_mi(Xg,std,Xt,a.k_next,a.min_dist,a.ls_corr)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'x':Xg[idx,0],'y':Xg[idx,1],'sigma':std[idx]}).to_csv(out_dir/'next_points.csv',index=False)
    print(f"[OK] {len(idx)} next points -> {out_dir/'next_points.csv'}")

if __name__=='__main__':
    main()
