from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os, argparse, json
import torch, gpytorch
import matplotlib.pyplot as plt
from src.models.dynamic_pde import simulate_plume
from src.models.pde_mean import PDEMean
from src.models.st_kernel import SeparableSTKernel
from src.models.exactgp_st import ExactGP_ST
from src.models.barrier import BarrierAttenuation
from src.utils.geo import load_first_polygon

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
    fields = simulate_plume(H=100,W=100,T=5, v=(0.2,0.05), kappa=args.kappa, dt=0.2, device=device)
    torch.manual_seed(0)
    n=120; T,H,W = fields.shape
    xs = torch.rand(n); ys = torch.rand(n); ts = torch.randint(0,T,(n,),dtype=torch.long)
    y  = fields[ts, (ys*(H-1)).long(), (xs*(W-1)).long()] + 0.05*torch.randn(n)
    X  = torch.stack([xs,ys, ts/(T-1)], dim=-1).to(device); y=y.to(device)

    mean_module = PDEMean(fields).to(device)
    space = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)).to(device)
    timek = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(-1,)).to(device)
    barrier=None
    if args.barrier_geojson and os.path.exists(args.barrier_geojson):
        try:
            poly=load_first_polygon(args.barrier_geojson)
            barrier=BarrierAttenuation(poly, gamma=args.barrier_gamma)
        except Exception:
            barrier=None
    covar_module = SeparableSTKernel(space, timek, barrier=barrier).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGP_ST(X, y, likelihood, mean_module, covar_module).to(device)

    model.train(); likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.08)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(100):
        opt.zero_grad(); out = model(X); loss = -mll(out, y); loss.backward(); opt.step()
        if (i+1)%25==0: print(f"[{i+1}] loss={loss.item():.4f}")

    model.eval(); likelihood.eval()
    grid=90; xs=torch.linspace(0,1,grid,device=device); ys=torch.linspace(0,1,grid,device=device)
    xx,yy=torch.meshgrid(xs,ys,indexing='xy'); tt=torch.full_like(xx, 0.6)
    Xt=torch.stack([xx.reshape(-1),yy.reshape(-1),tt.reshape(-1)],dim=-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred=likelihood(model(Xt)); mean=pred.mean.reshape(grid,grid).cpu(); std=pred.variance.sqrt().reshape(grid,grid).cpu()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Posterior mean"); plt.imshow(mean.T, origin='lower', extent=[0,1,0,1]); plt.colorbar()
    plt.subplot(1,2,2); plt.title("Posterior std");  plt.imshow(std.T,  origin='lower', extent=[0,1,0,1]);  plt.colorbar()
    plt.tight_layout(); plt.savefig("figures/mean_map.png", dpi=160); plt.close()
    plt.figure(figsize=(5,4)); plt.title("Posterior std (copy)"); plt.imshow(std.T, origin='lower', extent=[0,1,0,1]); plt.colorbar(); plt.tight_layout(); plt.savefig("figures/std_map.png", dpi=160); plt.close()

    metrics={"n_train":int(n),"note":"demo metrics (placeholder)"}
    with open("results/metrics.json","w") as f: json.dump(metrics,f,indent=2)
    print("Saved figures/mean_map.png, figures/std_map.png, results/metrics.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--use_pde_background", action="store_true")
    ap.add_argument("--barrier_geojson", type=str, default="")
    ap.add_argument("--barrier_gamma", type=float, default=4.0)
    ap.add_argument("--kappa", type=float, default=0.003)
    args=ap.parse_args()
    main(args)
