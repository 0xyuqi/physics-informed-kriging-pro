from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os, math
import torch, gpytorch
import matplotlib.pyplot as plt
from src.models.cokriging import AutoRegCoKriging

def make_data(nL=600, nH=60, seed=0):
    torch.manual_seed(seed)
    X = torch.rand(nL,2)
    f = torch.sin(2*math.pi*X[:,0]) + 0.5*torch.cos(2*math.pi*X[:,1])
    yL = f + 0.2*torch.randn_like(f)  # low-fidelity noisy
    perm = torch.randperm(nL)[:nH]
    XH = X[perm]; yH = (1.2*f[perm] + 0.05*torch.randn(nH))  # high-fidelity
    return X, yL, XH, yH

def main(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)

    XL, yL, XH, yH = make_data()
    XL, yL, XH, yH = XL.to(device), yL.to(device), XH.to(device), yH.to(device)

    ck = AutoRegCoKriging(); ck.fit(XL, yL, XH, yH, iters=120)
    # visualize over grid
    n=80; xs=torch.linspace(0,1,n,device=device); ys=torch.linspace(0,1,n,device=device)
    xx,yy=torch.meshgrid(xs,ys,indexing='xy'); Xg=torch.stack([xx.reshape(-1),yy.reshape(-1)],dim=-1)
    m,v=ck.predict(Xg); m=m.reshape(n,n).cpu(); s=v.sqrt().reshape(n,n).cpu()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Co-Kriging mean"); plt.imshow(m.T,origin='lower',extent=[0,1,0,1]); plt.colorbar()
    plt.subplot(1,2,2); plt.title("Co-Kriging std");  plt.imshow(s.T,origin='lower',extent=[0,1,0,1]); plt.colorbar()
    plt.tight_layout(); plt.savefig("figures/mean_cok.png",dpi=150); plt.close()
    print("Saved figures/mean_cok.png and std_cok.png")

if __name__ == "__main__":
    main()
