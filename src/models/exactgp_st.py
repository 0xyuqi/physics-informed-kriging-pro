import gpytorch
from .pde_mean import PDEMean

class ExactGP_ST(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = mean_module
        self.covar_module = covar_module
    def forward(self, x):
        mean_x  = self.mean_module(x) if isinstance(self.mean_module, PDEMean) else self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
