# Minimal import smoke test
import importlib
mods = [
    "src.models.dynamic_pde",
    "src.models.pde_mean",
    "src.models.st_kernel",
    "src.models.barrier",
    "src.models.dkl_model",
    "src.models.cokriging",
    "src.utils.sampling",
    "src.utils.geo",
]
for m in mods:
    importlib.import_module(m)
print("All core modules imported OK.")
