# scripts/run_dynamic_dkl.py
import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)  

import torch
import gpytorch
import matplotlib.pyplot as plt

from src.models.dynamic_pde import simulate_plume
from src.models.dkl_model import make_dkl
from src.models.cokriging import CoKrigingModel 

# from src.models.physics_background import PhysicsBackgroundMean

def synth_observations(fields, n_obs=250, noise=0.03, seed=42):
    torch.manual_seed(seed)
    T, H, W = fields.shape
    xs = torch.rand(n_obs)
    ys = torch.rand(n_obs)
    ts = torch.randint(0, T, (n_obs,), dtype=torch.long)
    vals = fields[ts, (ys*(H-1)).long(), (xs*(W-1)).long()]
    y = vals + noise * torch.randn_like(vals)
    X = torch.stack([xs, ys, ts / (T - 1)], dim=-1)
    return X, y

def main(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1 .  生成动态羽团
    fields = simulate_plume(H=128, W=128, T=25, v=(0.2,0.05), kappa=0.002, dt=0.2, device=device)
    torch.save(fields.cpu(), "data/dynamic_plume_fields.pt")

    # 2 . 采样观察数据
    Xtr, ytr = synth_observations(fields, n_obs=250, noise=0.03, seed=42)
    torch.save((Xtr.cpu(), ytr.cpu()), "data/dynamic_plume_obs.pt")

    # 3 .  DKL 模型
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    dkl, lik1 = make_dkl(in_dim=Xtr.shape[-1], train_x=Xtr, train_y=ytr)
    dkl, lik1 = dkl.to(device), lik1.to(device)

    optimizer = torch.optim.Adam(dkl.parameters(), lr=0.08)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik1, dkl)
    dkl.train(); lik1.train()
    for i in range(200):
        optimizer.zero_grad()
        out = dkl(Xtr)
        loss = -mll(out, ytr)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"[iter {i}] loss={loss.item():.4f}")

    dkl.eval(); lik1.eval()

    # 4 .  评估 on new points
    Xte, yte = synth_observations(fields, n_obs=120, noise=0.03, seed=999)
    Xte, yte = Xte.to(device), yte.to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik1(dkl(Xte))
        mean = pred.mean.cpu()
        var = pred.variance.cpu()

    torch.save((Xte.cpu(), yte.cpu(), mean, var), "data/dynamic_dkl_pred.pt")

    # 5 .  绘图：NLL & RMSE 对比（此示例无 baseline）
    rmse = torch.sqrt(torch.mean((mean - yte.cpu())**2)).item()
    nll = (0.5*(torch.log(2 * torch.pi * var) + (yte.cpu() - mean)**2/var)).mean().item()

    plt.figure(figsize=(4,3))
    plt.bar(["DKL"], [nll]); plt.title(f"NLL= {nll:.2f}")
    plt.savefig("figures/dkl_nll.png", dpi=150); plt.close()

    plt.figure(figsize=(4,3))
    plt.bar(["DKL"], [rmse]); plt.title(f"RMSE= {rmse:.2f}")
    plt.savefig("figures/dkl_rmse.png", dpi=150); plt.close()

    print("Figures & data saved to data/ and figures/")

if __name__ == "__main__":
    main()
