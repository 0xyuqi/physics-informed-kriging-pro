import torch, gpytorch, torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim:int, out_dim:int=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.SiLU(),
            nn.Linear(128,64), nn.SiLU(),
            nn.Linear(64,out_dim)
        )
    def forward(self, x): return self.net(x)

class DKLExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature: nn.Module):
    
        z = feature(train_x)
        super().__init__(z, train_y, likelihood)
        self.feature = feature
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=z.shape[-1])
        )
    def forward(self, x):
        z = self.feature(x)
        mean_x  = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def make_dkl(in_dim:int, train_x:torch.Tensor, train_y:torch.Tensor):
    feat = FeatureExtractor(in_dim, out_dim=32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = DKLExactGP(train_x, train_y, likelihood, feat)
    return model, likelihood
