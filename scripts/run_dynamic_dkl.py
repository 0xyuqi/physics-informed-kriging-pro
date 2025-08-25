from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os, math
import torch, gpytorch
import matplotlib.pyplot as plt
from src.models.dynamic_pde import simulate_plume
from src.models.dkl_model import make_dkl

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
    os.makedirs("figures", exist_ok=True); os.makedirs("data", exist_ok=True)

    fields = simulate_plume(H=100,W=100,T=25, v=(0.25,0.05), kappa=0.003, dt=0.2, device=device)
    Xtr, ytr = synth_observations(fields, n_obs=250, noise=0.03, seed=42)
    Xtr, ytr = Xtr.to(device), ytr.to(device)

    dkl, lik = make_dkl(in_dim=Xtr.shape[-1], train_x=Xtr, train_y=ytr)
    dkl, lik = dkl.to(device), lik.to(device)

    dkl.train(); lik.train()
    opt = torch.optim.Adam(dkl.parameters(), lr=0.08)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, dkl)
    for i in range(160):
        opt.zero_grad(); out = dkl(Xtr); loss = -mll(out, ytr); loss.backward(); opt.step()
        if i % 40 == 0: print(f"[iter {i}] loss={loss.item():.4f}")

    dkl.eval(); lik.eval()
    Xte, yte = synth_observations(fields, n_obs=120, noise=0.03, seed=999)
    Xte, yte = Xte.to(device), yte.to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(dkl(Xte))
        mean, var = pred.mean.cpu(), pred.variance.cpu()

    rmse = torch.sqrt(torch.mean((mean - yte.cpu())**2)).item()
    nll = (0.5*(torch.log(2*math.pi*var) + (yte.cpu()-mean)**2/var)).mean().item()

    plt.figure(figsize=(4,3)); plt.title(f"NLL = {nll:.2f}")
    plt.bar(["DKL"], [nll]); plt.tight_layout(); plt.savefig("figures/dkl_nll.png", dpi=150); plt.close()
    plt.figure(figsize=(4,3)); plt.title(f"RMSE = {rmse:.2f}")
    plt.bar(["DKL"], [rmse]); plt.tight_layout(); plt.savefig("figures/dkl_rmse.png", dpi=150); plt.close()

    print("Saved figures/dkl_*.png")

if __name__ == "__main__":
    main()
