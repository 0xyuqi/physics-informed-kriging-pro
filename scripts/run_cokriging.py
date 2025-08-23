"""Co-Kriging (Kennedy–O'Hagan) / 低保真+高保真融合."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

def save_heatmap(df_grid, Z, pts_df, title, path):
    nx=int(df_grid['i'].max()+1); ny=int(df_grid['j'].max()+1)
    plt.figure()
    plt.imshow(Z.reshape(ny,nx), origin='lower',
               extent=[df_grid['x'].min(),df_grid['x'].max(),df_grid['y'].min(),df_grid['y'].max()])
    if pts_df is not None:
        plt.scatter(pts_df['x'], pts_df['y'], s=14, alpha=0.85, edgecolor='k', linewidths=0.2)
    plt.title(title); plt.colorbar(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',type=str,default='data'); ap.add_argument('--out_dir',type=str,default='figures')
    ap.add_argument('--n_lowfit',type=int,default=800); ap.add_argument('--lf_length',type=float,default=20.0); ap.add_argument('--lf_noise',type=float,default=1e-3)
    ap.add_argument('--hf_length',type=float,default=15.0); ap.add_argument('--hf_noise',type=float,default=1e-3)
    a=ap.parse_args()

    data_dir, out_dir = Path(a.data_dir), Path(a.out_dir)
    hi=pd.read_csv(data_dir/'synth_points.csv'); lo=pd.read_csv(data_dir/'proxy_points.csv'); grid=pd.read_csv(data_dir/'grid_coords.csv')

    if lo.shape[0] > a.n_lowfit: lo=lo.sample(a.n_lowfit, random_state=0).reset_index(drop=True)
    X_L=lo[['x','y']].to_numpy(); y_L=lo['proxy'].to_numpy()
    gp_L=GaussianProcessRegressor(kernel=C(1.0,(1e-3,1e3))*RBF(a.lf_length,(1e-2,1e3))+WhiteKernel(a.lf_noise),
                                  alpha=1e-6, normalize_y=True, random_state=0)
    gp_L.fit(X_L,y_L)

    X_H=hi[['x','y']].to_numpy(); y_H=hi['z'].to_numpy()
    mu_L_H,_=gp_L.predict(X_H,return_std=True)
    rho=float(np.dot(mu_L_H,y_H)/max(np.dot(mu_L_H,mu_L_H),1e-12))

    res=y_H - rho*mu_L_H
    gp_D=GaussianProcessRegressor(kernel=C(1.0,(1e-3,1e3))*RBF(a.hf_length,(1e-2,1e3))+WhiteKernel(a.hf_noise),
                                  alpha=1e-6, normalize_y=True, random_state=0)
    gp_D.fit(X_H,res)

    Xg=grid[['x','y']].to_numpy()
    mu_L_G,std_L_G=gp_L.predict(Xg,return_std=True); mu_D_G,std_D_G=gp_D.predict(Xg,return_std=True)
    mu_H_G=rho*mu_L_G + mu_D_G; std_H_G=np.sqrt((rho**2)*(std_L_G**2) + (std_D_G**2))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'x':grid['x'],'y':grid['y'],'mean':mu_H_G,'std_y':std_H_G}).to_csv(data_dir/'grid_pred_cok.csv',index=False)
    save_heatmap(grid, mu_H_G, hi, 'Co-Kriging Mean', out_dir/'mean_cok.png')
    save_heatmap(grid, std_H_G, hi, 'Co-Kriging Std',  out_dir/'std_cok.png')
    print(f"[OK] rho={rho:.3f}; outputs saved.")

if __name__=='__main__':
    main()
