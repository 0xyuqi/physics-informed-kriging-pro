import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.models.cokriging import CoKrigingAR1, CKConfig
from src.design.acquisition import select_by_variance, select_by_a_optimal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--k_next", type=int, default=8)
    ap.add_argument("--strategy", choices=["var", "aopt"], default="var")
    ap.add_argument("--min_dist", type=float, default=0.0)
    ap.add_argument("--out_csv", default="figures/next_points.csv")
    args = ap.parse_args()

    obs_path = Path(args.data_dir) / "obs_points.csv"
    grid_path = Path(args.data_dir) / "grid.csv"
    proxy_path = Path(args.data_dir) / "proxy_points.csv"
    if not (obs_path.exists() and grid_path.exists() and proxy_path.exists()):
        raise FileNotFoundError("缺少 obs_points.csv / grid.csv / proxy_points.csv，请先生成数据并跑一次模型。")

    obs = pd.read_csv(obs_path)
    grid = pd.read_csv(grid_path)
    proxy = pd.read_csv(proxy_path)

    cfg = CKConfig()
    model = CoKrigingAR1(cfg).fit(
        X_L=proxy[["x", "y"]].values,
        y_L=proxy["z"].values,
        X_H=obs[["x", "y"]].values,
        y_H=obs["z"].values,
    )

    Xcand = grid[["x", "y"]].values
    if args.strategy == "var":
        idx = select_by_variance(model, Xcand, k=args.k_next, min_dist=args.min_dist)
    else:
      
        idx = select_by_a_optimal(model, Xcand, grid_eval=Xcand, k=args.k_next, min_dist=args.min_dist)

    out = grid.iloc[idx].copy()
    out["std_now"] = model.predict(out[["x", "y"]].values, return_std=True)[1]
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("Next candidates saved to:", args.out_csv)


if __name__ == "__main__":
    main()
