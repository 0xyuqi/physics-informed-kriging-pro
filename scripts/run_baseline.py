"""Advanced baseline（PDE + 非平稳 + 屏障 / bilingual）"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from shapely.geometry import shape, Polygon
from scipy.stats import norm

from src.models.physics_background import solve_background
from src.models.advanced_gp import AdvancedAnisotropicGP

def crps_gaussian(y, mu, s):
    s = np.maximum(np.asarray(s,float), 1e-12)
    z = (np.asarray(y)-np.asarray(mu))/s
    return s*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))

def save_heatmap(Z, pts_df, grid_df, title, path):
    plt.figure()
    plt.imshow(Z, origin='lower', extent=[grid_df['x'].min(),grid_df['x'].max(),grid_df['y'].min(),grid_df['y'].max()])
    if pts_df is not None:
        plt.scatter(pts_df['x'], pts_df['y'], s=14, alpha=0.85, edgecolor='k', linewidths=0.2)
    plt.title(title); plt.colorbar(); plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(path, dpi=160); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',type=str,default='data'); ap.add_argument('--out_dir',type=str,default='figures')
    ap.add_argument('--length_parallel',type=float,default=30.0); ap.add_argument('--length_cross',type=float,default=8.0)
    ap.add_argument('--rq_alpha',type=float,default=0.5); ap.add_argument('--alpha_jitter',type=float,default=1e-6)
    ap.add_argument('--noise_level',type=float,default=1e-3); ap.add_argument('--no_opt',action='store_true'); ap.add_argument('--n_restarts',type=int,default=2)
    ap.add_argument('--use_pde_background',action='store_true'); ap.add_argument('--kappa',type=float,default=2.0); ap.add_argument('--c_in',type=float,default=0.0)
    ap.add_argument('--source_amp',type=float,default=0.0); ap.add_argument('--source_x',type=float,default=50.0); ap.add_argument('--source_y',type=float,default=50.0)
    ap.add_argument('--barrier_geojson',type=str,default=''); ap.add_argument('--barrier_gamma',type=float,default=5.0)
    ap.add_argument('--nonstat_boost_along',type=float,default=20.0); ap.add_argument('--nonstat_boost_cross',type=float,default=5.0); ap.add_argument('--nonstat_tau',type=float,default=8.0)
    args=ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    pts = pd.read_csv(data_dir/'synth_points.csv'); grid = pd.read_csv(data_dir/'grid_coords.csv')
    meta = json.loads((data_dir/'flow_meta.json').read_text()); vx,vy = float(meta['vx']), float(meta['vy'])

    coast_poly=None
    if args.barrier_geojson and Path(args.barrier_geojson).exists():
        gj=json.loads(Path(args.barrier_geojson).read_text()); coast_poly=shape(gj['features'][0]['geometry'])

    grid_xy = np.stack([grid['x'].values, grid['y'].values], axis=1).reshape(int(grid['j'].max()+1), int(grid['i'].max()+1), 2)
    pde_field=None
    if args.use_pde_background:
        pde_field = solve_background(grid, vx, vy, kappa=args.kappa, c_in=args.c_in, coast_polygon=coast_poly,
                                     source_amp=args.source_amp, source_xy=(args.source_x,args.source_y))

    model = AdvancedAnisotropicGP(vx, vy, lp=args.length_parallel, lc=args.length_cross, rq_alpha=args.rq_alpha,
                                  noise_level=args.noise_level, learn_noise=False,
                                  mean_mode=('pde' if args.use_pde_background else 'poly'),
                                  use_quadratic_trend=True, grid_xy=(grid_xy if pde_field is not None else None),
                                  pde_field=pde_field, coast_polygon=coast_poly, barrier_gamma=args.barrier_gamma,
                                  nonstat_boost=(args.nonstat_boost_along,args.nonstat_boost_cross),
                                  nonstat_tau=args.nonstat_tau, alpha_jitter=args.alpha_jitter,
                                  optimize=(not args.no_opt), n_restarts=args.n_restarts)
    X=pts[['x','y']].to_numpy(); y=pts['z'].to_numpy(); model.fit(X,y)

    Xg=grid[['x','y']].to_numpy(); mu,std=model.predict(Xg,return_std=True)
    nx=int(grid['i'].max()+1); ny=int(grid['j'].max()+1); Zm=mu.reshape(ny,nx); Zs=std.reshape(ny,nx)
    save_heatmap(Zm, pts, grid, 'Mean (PDE+Nonstationary+Barrier)', out_dir/'mean_map.png')
    save_heatmap(Zs, pts, grid, 'Std (PDE+Nonstationary+Barrier)',  out_dir/'std_map.png')

    # 10-fold CV
    from sklearn.model_selection import KFold
    kf=KFold(n_splits=min(10,len(X))); preds=[]; sigs=[]; obs=[]
    for tr,te in kf.split(X):
        m2=AdvancedAnisotropicGP(vx,vy,lp=args.length_parallel, lc=args.length_cross, rq_alpha=args.rq_alpha,
                                 noise_level=args.noise_level, learn_noise=False,
                                 mean_mode=('pde' if args.use_pde_background else 'poly'),
                                 use_quadratic_trend=True, grid_xy=(grid_xy if pde_field is not None else None),
                                 pde_field=pde_field, coast_polygon=coast_poly, barrier_gamma=args.barrier_gamma,
                                 nonstat_boost=(args.nonstat_boost_along,args.nonstat_boost_cross),
                                 nonstat_tau=args.nonstat_tau, alpha_jitter=args.alpha_jitter,
                                 optimize=False, n_restarts=0)
        m2.fit(X[tr], y[tr]); mu_te,std_te=m2.predict(X[te],return_std=True)
        preds.append(mu_te.ravel()); sigs.append(std_te.ravel()); obs.append(y[te].ravel())
    yhat=np.concatenate(preds); ysig=np.concatenate(sigs); ytrue=np.concatenate(obs)
    mae=float(np.mean(np.abs(yhat-ytrue))); rmse=float(np.sqrt(np.mean((yhat-ytrue)**2))); crps=float(np.mean(crps_gaussian(ytrue,yhat,ysig)))
    (out_dir/'metrics.json').write_text(json.dumps({'MAE':mae,'RMSE':rmse,'CRPS':crps},indent=2))
    pd.DataFrame({'x':grid['x'],'y':grid['y'],'mean':mu,'std_y':std}).to_csv(data_dir/'grid_pred.csv',index=False)
    print(f"[10-fold] MAE={mae:.4f} RMSE={rmse:.4f} CRPS={crps:.4f}")
    print(f"[OK] figures -> {out_dir} ; predictions -> {data_dir/'grid_pred.csv'}")

if __name__=='__main__':
    main()
