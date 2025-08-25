from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os, math
import torch, gpytorch
import matplotlib.pyplot as plt
from src.models.dkl_model import make_dkl

def friedman1(n, d=10, noise=0.1, seed=0):
    torch.manual_seed(seed)
    X = torch.rand(n, d)
    y = 10*torch.sin(math.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4]
    y = y + noise*torch.randn_like(y)
    return X, y

class PlainExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

def train_exact(model, likelihood, X, y, iters=150, lr=0.08):
    model.train(); likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(1, iters+1):
        opt.zero_grad(); out = model(X); loss = -mll(out, y); loss.backward(); opt.step()
        if i%50==0: print(f"[{i}] loss={loss.item():.3f}")
    model.eval(); likelihood.eval()

def eval_nll_rmse(fwd, likelihood, X, y):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(fwd(X))
        mean = pred.mean; var = pred.variance
        nll  = (0.5*(torch.log(2*math.pi*var) + (y-mean)**2/var)).mean().item()
        rmse = torch.sqrt(torch.mean((mean-y)**2)).item()
    return nll, rmse

def main(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("figures", exist_ok=True)
    X, y = friedman1(400, d=20, noise=0.2, seed=0)
    Xtr, Ytr = X[:300].to(device), y[:300].to(device)
    Xte, Yte = X[300:].to(device), y[300:].to(device)

    lik0 = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp0  = PlainExactGP(Xtr, Ytr, lik0).to(device)
    train_exact(gp0, lik0, Xtr, Ytr)

    dkl, lik1 = make_dkl(in_dim=Xtr.shape[-1], train_x=Xtr, train_y=Ytr)
    dkl, lik1 = dkl.to(device), lik1.to(device)
    train_exact(dkl, lik1, Xtr, Ytr)

    nll0, rmse0 = eval_nll_rmse(lambda x: gp0.forward(x), lik0, Xte, Yte)
    nll1, rmse1 = eval_nll_rmse(lambda x: dkl.forward(x), lik1, Xte, Yte)

    print(f"Plain GP: NLL={nll0:.3f} RMSE={rmse0:.3f}")
    print(f"DKL     : NLL={nll1:.3f} RMSE={rmse1:.3f}")

    plt.figure(figsize=(6,4)); plt.title("NLL (lower is better)"); plt.bar(["Plain GP","DKL"], [nll0,nll1]); plt.tight_layout(); plt.savefig("figures/dkl_nll.png", dpi=160); plt.close()
    plt.figure(figsize=(6,4)); plt.title("RMSE (lower is better)"); plt.bar(["Plain GP","DKL"], [rmse0,rmse1]); plt.tight_layout(); plt.savefig("figures/dkl_rmse.png", dpi=160); plt.close()
    print("Saved figures/dkl_nll.png and figures/dkl_rmse.png")

if __name__ == "__main__":
    main()
