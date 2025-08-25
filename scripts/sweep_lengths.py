from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os, argparse, itertools
import pandas as pd
import matplotlib.pyplot as plt

def main(lp_list, lc_list):
    os.makedirs("summary", exist_ok=True)
    rows=[]
    for lp in lp_list:
        for lc in lc_list:
            rmse = ((lp-30)**2/900 + (lc-10)**2/100) + 0.2  # convex bowl
            rows.append({"lp":lp,"lc":lc,"rmse":rmse})
    df=pd.DataFrame(rows)
    df.to_csv("summary/all_metrics.csv", index=False)
    piv = df.pivot(index="lc", columns="lp", values="rmse")
    plt.figure(figsize=(6,4)); plt.title("RMSE sweep (synthetic)")
    im=plt.imshow(piv.values, origin="lower", extent=[min(lp_list),max(lp_list),min(lc_list),max(lc_list)], aspect="auto")
    plt.xlabel("length_parallel (lp)"); plt.ylabel("length_cross (lc)"); plt.colorbar(im,label="rmse")
    plt.tight_layout(); plt.savefig("summary/heatmap.png", dpi=160); plt.close()
    print("Saved summary/heatmap.png and all_metrics.csv")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--lp_list", nargs="+", type=float, default=[20,30,40])
    ap.add_argument("--lc_list", nargs="+", type=float, default=[6,8,12])
    args=ap.parse_args()
    main(args.lp_list, args.lc_list)
