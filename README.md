# Physics-Informed Kriging


---

## 亮点 Highlights

* 物理先验：稳态对流–扩散方程在网格上求解，双线性插值作为 GP 均值并线性校准
  Steady advection–diffusion PDE solved on grid → bilinear interpolation as mean with linear calibration
* 核函数：RBF(ARD, 流向/横向) + RationalQuadratic(多尺度) + 非平稳 Gibbs 风格核 × 海岸屏障核 + White 噪声
  RBF(ARD) + RQ + non-stationary Gibbs-like × barrier + white noise
* 低成本代理融合：Kennedy–O’Hagan 自回归 Co-Kriging（ρ·f\_L + δGP）
  Autoregressive co-kriging fusion of low- and high-fidelity signals
* 主动采样：互信息 logdet 贪心 + 最小距离约束
  MI-greedy with minimum distance constraint
* 评估完整：10-fold 近似 LOO 的 MAE / RMSE / CRPS；超参数扫描热力图
  10-fold CV for MAE/RMSE/CRPS; sweep heatmap

---

## 目录结构 Project Tree

```
physics-informed-kriging-pro/
├─ requirements.txt                 # 固定依赖版本 / pinned deps
├─ src/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  └─ geo.py                    # 坐标旋转：x,y ↔ 沿流/横流
│  └─ models/
│     ├─ __init__.py
│     ├─ nonstationary_kernels.py  # 非平稳(Gibbs风格)旋转核
│     ├─ physics_background.py     # PDE 背景均值（对流–扩散）
│     └─ advanced_gp.py            # GP 封装（核+均值+屏障）
├─ scripts/
│  ├─ generate_synth.py            # 合成数据 + 低价代理
│  ├─ run_baseline.py              # 终极基线（图+指标+预测表）
│  ├─ run_cokriging.py             # Co-Kriging 融合
│  ├─ select_next_samples.py       # 主动采样点选择
│  └─ sweep_lengths.py             # (lp,lc) 扫描 → 热力图
├─ data/
│  ├─ synth_points.csv            
│  ├─ grid_coords.csv
│  ├─ flow_meta.json
│  ├─ proxy_grid.csv
│  ├─ proxy_points.csv
│  └─ malaysia_coast_example.geojson  # 占位海岸线，多边形陆地
├─ figures/.gitkeep
└─ summary/.gitkeep
```

---

## 环境 Environment

依赖见 `requirements.txt`。
Recommended Python 3.10–3.11 for stable wheels on Windows.

### Windows PowerShell

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
# 若遇“脚本被禁用”：以管理员 PowerShell 执行
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -r requirements.txt
```

### macOS / Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 开始 Quickstart

仓库自带 `data/`，可以直接跑，不需要先生成数据。
Data is pre-baked; you can run straight away.

```powershell
# 基线（启用 PDE 背景、屏障核；示例海岸线）
python scripts\\run_baseline.py --use_pde_background --barrier_geojson data\\malaysia_coast_example.geojson
```

产物 Artifacts：

* `figures/mean_map.png`：后验均值 Posterior mean
* `figures/std_map.png`：后验标准差 Uncertainty
* `figures/metrics.json`：MAE/RMSE/CRPS（10-fold）
* `data/grid_pred.csv`：全网格均值与不确定性

---

## 合成数据与代理 Synthetic Data and Proxy

重新生成（可改样本数、网格、流速等）：

```powershell
python scripts\\generate_synth.py --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42
```

同时会输出：

* `proxy_grid.csv` 与 `proxy_points.csv`（模拟“遥感浊度指数”等低成本代理），用于 Co-Kriging。

---

## 真实海岸线替换 Real Malaysia Coastline

将海岸线 GeoJSON 放入 `data/`（多边形表示**陆地**）。运行时指定：

```powershell
python scripts\\run_baseline.py --use_pde_background --barrier_geojson data\\your_malaysia_coast.geojson
```

屏障核会检测任意两点连线是否穿过陆地，多次相交会指数级抑制相干性，避免跨陆“穿透预测”。

---

## 主动采样 Active Sampling

基于互信息 logdet 贪心 + 最小距离约束，给出“下一轮采样点”：

```powershell
python scripts\\select_next_samples.py --k_next 8 --min_dist 3.0
```

输出 `figures/next_points.csv`，包含建议坐标与当前不确定性。

---

## 低价代理融合 Co-Kriging

Kennedy–O’Hagan 自回归：`z(x) = ρ·f_L(x) + δ(x)`，其中 `f_L` 由低价代理拟合。
Autoregressive co-kriging: `z = rho * f_L + delta`.

```powershell
python scripts\\run_cokriging.py --n_lowfit 800 --lf_length 20 --hf_length 15
```

输出：

* `figures/mean_cok.png`、`figures/std_cok.png`
* `data/grid_pred_cok.csv`

---

## 参数扫描与热力图 Sweep + Heatmap

对流向/横向长度尺度 `(lp, lc)` 网格扫描，生成 RMSE 热力图：

```powershell
python scripts\\sweep_lengths.py --lp_list 20 30 40 --lc_list 6 8 12 --use_pde_background
```

输出：

* `summary/all_metrics.csv`
* `summary/heatmap.png`

---

## 常用参数 Common Flags

`run_baseline.py`：

* `--use_pde_background` 使用 PDE 背景均值（否则为沿流/横流二次多项式趋势）
* `--kappa`、`--c_in`、`--source_amp`、`--source_x`、`--source_y` 控制 PDE 扩散系数、入流浓度与源项
* `--barrier_geojson`、`--barrier_gamma` 屏障核与穿越惩罚
* `--length_parallel`、`--length_cross`、`--rq_alpha` 核长度与 RQ 形状参数
* `--nonstat_boost_along`、`--nonstat_boost_cross`、`--nonstat_tau` 非平稳长度随“离岸距离”的增强幅度与尺度
* `--no_opt`、`--n_restarts` 关闭/开启超参优化与重启次数


---

## 复现实验 Reproduce in One Go

```powershell
# 生成数据
python scripts\\generate_synth.py --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42

# 终极基线（启用 PDE + 海岸屏障 + 非平稳核）
python scripts\\run_baseline.py --use_pde_background --barrier_geojson data\\malaysia_coast_example.geojson

# 主动采样
python scripts\\select_next_samples.py --k_next 8

# Co-Kriging 融合
python scripts\\run_cokriging.py

# 扫描热力图
python scripts\\sweep_lengths.py --lp_list 20 30 40 --lc_list 6 8 12 --use_pde_background

