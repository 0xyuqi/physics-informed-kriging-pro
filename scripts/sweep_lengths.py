"""(lp, lc) 参数扫描 → CSV + RMSE 热力图 / Parameter sweep."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import json as jsonlib
from src.models.advanced_gp import AdvancedAnisotropicGP

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',type=str,default='data'); ap.add_argument('--out_dir',type=str,default='summary')
    ap.add_argument('--lp_list',type=float,nargs='+',default=[20,30,40]); ap.add_argument('--lc_list',type=float,nargs='+',default=[6,8,12])
    ap.add_argument('--rq_alpha',type=float,default=0.5); ap.add_argument('--use_pde_background',action='store_true')
    a=ap.parse_args()

    data_dir, out_dir = Path(a.data_dir), Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pts=pd.read_csv(data_dir/'synth_points.csv'); meta=jsonlib.loads((data_dir/'flow_meta.json').read_text())
    vx,vy=float(meta['vx']), float(meta['vy']); X=pts[['x','y']].to_numpy(); y=pts['z'].to_numpy()

    rows=[]
    for lp in a.lp_list:
        for lc in a.lc_list:
            kf=KFold(n_splits=min(10,len(X))); preds=[]; obs=[]
            for tr,te in kf.split(X):
                m=AdvancedAnisotropicGP(vx,vy,lp=lp,lc=lc,rq_alpha=a.rq_alpha,
                                        mean_mode=('pde' if a.use_pde_background else 'poly'),
                                        optimize=False, n_restarts=0, learn_noise=False)
                m.fit(X[tr],y[tr]); mu,_=m.predict(X[te],return_std=True)
                preds.append(mu.ravel()); obs.append(y[te].ravel())
            yhat=np.concatenate(preds); ytrue=np.concatenate(obs)
            mae=float(np.mean(np.abs(yhat-ytrue))); rmse=float(np.sqrt(np.mean((yhat-ytrue)**2)))
            rows.append({'lp':lp,'lc':lc,'MAE':mae,'RMSE':rmse})
            print(f"[sweep] lp={lp} lc={lc} -> RMSE={rmse:.4f}, MAE={mae:.4f}")
    df=pd.DataFrame(rows); df.to_csv(out_dir/'all_metrics.csv',index=False)
    P=df.pivot(index='lc',columns='lp',values='RMSE')
    plt.figure(); im=plt.imshow(P.values,origin='lower',aspect='auto')
    plt.xticks(range(len(P.columns)),P.columns); plt.yticks(range(len(P.index)),P.index)
    plt.xlabel('length_parallel (lp)'); plt.ylabel('length_cross (lc)')
    plt.title('RMSE heatmap (advanced GP)'); plt.colorbar(im)
    import numpy as np as _np
    for i,lc in enumerate(P.index):
        for j,lp in enumerate(P.columns):
            v=P.loc[lc,lp]; 
            if _np.isfinite(v): plt.text(j,i,f"{v:.3f}",ha='center',va='center',fontsize=8)
    plt.tight_layout(); plt.savefig(out_dir/'heatmap.png',dpi=160); plt.close()
    print(f"[OK] summary saved -> {out_dir}")

if __name__=='__main__':
    main()
