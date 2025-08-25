import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.cokriging import CoKrigingAR1, CKConfig


def load_low_high_points(data_dir: str):
   
    p_low = Path(data_dir) / "proxy_points.csv"
    p_high = Path(data_dir) / "obs_points.csv"
    if not p_low.exists() or not p_high.exists():
        raise FileNotFoundError(f"Expect {p_low} and {p_high}. 请先运行 scripts/generate_synth.py 生成示例数据。")
    dfL = pd.read_csv(p_low)
    dfH = pd.read_csv(p_high)
    return dfL, dfH


def load_grid(data_dir: str):
    p = Path(data_dir) / "grid.csv"
    if not p.exists():
        raise FileNotFoundError(f"Expect {p}. 请先运行 scripts/generate_synth.py 生成网格。")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="数据目录")
    ap.add_argument("--out_dir", default="figures", help="图片输出目录")
    ap.add_argument("--n_restarts", type=int, default=2)
    ap.add_argument("--lf_length", type=float, nargs=2, default=[20.0, 8.0])
    ap.add_argument("--hf_length", type=float, nargs=2, default=[15.0, 6.0])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dfL, dfH = load_low_high_points(args.data_dir)
    grid = load_grid(args.data_dir)

    X_L = dfL[["x", "y"]].values
    y_L = dfL["z"].values
    X_H = dfH[["x", "y"]].values
    y_H = dfH["z"].values
    Xg = grid[["x", "y"]].values

    cfg = CKConfig(lf_length=tuple(args.lf_length), hf_length=tuple(args.hf_length), n_restarts=args.n_restarts)
    model = CoKrigingAR1(cfg).fit(X_L, y_L, X_H, y_H)

    mean, std = model.predict(Xg, return_std=True)
    grid_out = grid.copy()
    grid_out["z_pred"] = mean
    grid_out["z_std"] = std
    grid_out.to_csv("data/grid_pred_cok.csv", index=False)

    if {"i", "j"}.issubset(grid.columns):
        ni, nj = int(grid["i"].max()) + 1, int(grid["j"].max()) + 1
        Zm = grid_out.pivot_table(index="i", columns="j", values="z_pred").values
        Zs = grid_out.pivot_table(index="i", columns="j", values="z_std").values
        plt.figure(figsize=(5, 4))
        plt.title("Co-Kriging mean")
        plt.imshow(Zm, origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "mean_cok.png"), dpi=200)

        plt.figure(figsize=(5, 4))
        plt.title("Co-Kriging std")
        plt.imshow(Zs, origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "std_cok.png"), dpi=200)

    print("Saved:", "figures/mean_cok.png", "figures/std_cok.png", "data/grid_pred_cok.csv")


if __name__ == "__main__":
    main()

