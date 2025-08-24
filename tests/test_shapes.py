import torch, gpytorch
from pik_ext.physics.time_pde import simulate_plume
from pik_ext.gp.pde_mean import PDEMean
from pik_ext.gp.st_kernel import SeparableSTKernel

def test_pde_mean_sampling():
    fields = simulate_plume(H=16,W=16,T=6, v=(0.1,0.05), kappa=0.005, dt=0.3, device="cpu")
    mean = PDEMean(fields)
    X = torch.rand(10,3)
    y = mean(X)
    assert y.shape == (10,)

def test_separable_kernel_shapes():
    space = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    timek = gpytorch.kernels.MaternKernel(nu=0.5, active_dims=(-1,))
    K = SeparableSTKernel(space, timek)
    X = torch.rand(20,3)
    Kxx = K(X)
    assert Kxx.shape == (20,20)
