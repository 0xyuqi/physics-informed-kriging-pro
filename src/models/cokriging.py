import torch, gpytorch
from typing import Tuple

class PlainExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class AutoRegCoKriging:
    """ Autoregressive co-kriging (simple 2-stage impl):
        1) Fit low-fidelity GP:  f_L ~ GP
        2) Estimate rho via least-squares on HF subset
        3) Fit residual GP: y_H - rho * f_L_pred ~ GP
    """
    def __init__(self): pass
    def fit(self, XL, yL, XH, yH, iters:int=150, lr:float=0.08):
        device = XL.device
        # 1) low-fidelity GP
        self.likL = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.gpL  = PlainExactGP(XL, yL, self.likL).to(device)
        self._train_exact(self.gpL, self.likL, XL, yL, iters, lr)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            fL_H = self.likL(self.gpL(XH)).mean
        # 2) rho by least squares
        self.rho = ( (fL_H*yH).sum() / (fL_H*fL_H).sum() ).detach()
        # 3) residual GP
        r = yH - self.rho * fL_H
        self.likR = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.gpR  = PlainExactGP(XH, r, self.likR).to(device)
        self._train_exact(self.gpR, self.likR, XH, r, iters, lr)
        self.XH = XH  # cache
    def predict(self, X:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mL = self.likL(self.gpL(X)).mean
            mR = self.likR(self.gpR(X)).mean
        mean = self.rho * mL + mR
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            vL = self.likL(self.gpL(X)).variance
            vR = self.likR(self.gpR(X)).variance
        var = (self.rho**2)*vL + vR  # optimistic sum
        return mean, var
    def _train_exact(self, model, likelihood, X, y, iters, lr):
        model.train(); likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(iters):
            opt.zero_grad(); out = model(X); loss = -mll(out, y); loss.backward(); opt.step()
