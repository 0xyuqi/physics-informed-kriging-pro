from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os, argparse
import torch
import pandas as pd
from src.utils.sampling import greedy_epv_with_min_dist

def main(k_next:int, min_dist:float):
    os.makedirs("figures", exist_ok=True)
    # Simple demo: random candidate grid with synthetic variance
    n=2000
    Xcand = torch.rand(n,2)
    var = torch.exp(-((Xcand[:,0]-0.5)**2+(Xcand[:,1]-0.5)**2)/0.06) + 0.1*torch.rand(n)
    picks = greedy_epv_with_min_dist(Xcand, var, k=k_next, dmin=min_dist)
    df = pd.DataFrame({"x":Xcand[picks,0].cpu().numpy(),"y":Xcand[picks,1].cpu().numpy(),"var":var[picks].cpu().numpy()})
    df.to_csv("figures/next_points.csv", index=False)
    print("Saved figures/next_points.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_next", type=int, default=8)
    ap.add_argument("--min_dist", type=float, default=0.1)
    args = ap.parse_args()
    main(args.k_next, args.min_dist)
