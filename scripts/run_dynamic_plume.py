import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
for p in [ROOT, os.path.join(ROOT, "src")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch, gpytorch, os
import matplotlib.pyplot as plt
from src.models.dynamic_pde import simulate_plume
from src.models.pde_mean import PDEMean
from src.models.st_kernel import SeparableSTKernel
from src.models.exactgp_st import ExactGP_ST

def synth_observations(fields, n_obs=250, noise=0.03, seed=42):
    torch.manual_seed(seed)
    T,H,W = fields.shape
    xs = torch.rand(n_obs); ys = torch.rand(n_obs)
    ts = torch.randint(0,T,(n_obs,), dtype=torch.long)
    vals = fields[ts, (ys*(H-1)).long(), (xs*(W-1)).long()]
    y = vals + noise*torch.randn_like(vals)
    X = torch.stack([xs,ys, ts/(T-1)], dim=-1)
    return X, y

def main(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("figures", exist_ok=True)

    fields = simulate_plume(H=128,W=128,T=25, v=(0.2,0.05), kappa=0.002, dt=0.2, device=device)
    Xtr, ytr = synth_observations(fields, n_obs=250, noise=0.03, seed=42)
    Xtr, ytr = Xtr.to(device), ytr.to(device)

    mean_module = PDEMean(fields, xlim=(0,1), ylim=(0,1), tlim=(0,1)).to(device)
    space = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)).to(device)
    timek = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(-1,)).to(device)
    covar_module = SeparableSTKernel(space, timek).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGP_ST(Xtr, ytr, likelihood, mean_module, covar_module).to(device)

    model.train(); likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.08)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(1, 160+1):
        opt.zero_grad()
        out = model(Xtr)
        loss = -mll(out, ytr)
        loss.backward(); opt.step()
        if i%40==0: print(f"[iter {i}] loss={loss.item():.4f}")

    model.eval(); likelihood.eval()
    t_eval = 0.6; grid_n = 70
    xs = torch.linspace(0,1,grid_n, device=device)
    ys = torch.linspace(0,1,grid_n, device=device)
    xx,yy = torch.meshgrid(xs,ys, indexing='xy'); tt = torch.full_like(xx, t_eval)
    Xt = torch.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], dim=-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(Xt))
        mean = pred.mean.reshape(grid_n,grid_n).cpu()
        std  = pred.variance.sqrt().reshape(grid_n,grid_n).cpu()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Posterior mean t=0.6"); plt.imshow(mean.T, origin='lower', extent=[0,1,0,1]); plt.colorbar()
    plt.subplot(1,2,2); plt.title("Posterior std t=0.6");  plt.imshow(std.T,  origin='lower', extent=[0,1,0,1]); plt.colorbar()
    plt.tight_layout(); plt.savefig("figures/plume_posterior.png", dpi=160); plt.close()
    print("Saved figures/plume_posterior.png")

if __name__ == "__main__":
    main()
