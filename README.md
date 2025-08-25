# Physics-Informed Kriging (PIK)

> 高可信空间插值与**最优下一步采样**，面向水质热点等环境场景  
> High-confidence spatial interpolation + **next-best sampling** for environmental hotspots

---

## ✨ Highlights / 亮点

- **Physics prior 物理先验**  
  EN: Steady advection–diffusion PDE solved on grid ⇒ bilinear interpolation as GP mean + linear calibration  
  CN: 稳态对流–扩散方程在网格上求解，双线性插值作为 GP **均值**并线性校准

- **Kernels 核函数**  
  EN: RBF (ARD along/cross-flow) + RationalQuadratic (multi-scale) + non-stationary Gibbs-like × coastline **barrier** + white noise  
  CN: RBF（ARD，流向/横向）+ RQ（多尺度）+ **非平稳** Gibbs 风格核 × **海岸屏障核** + 白噪声

- **Low-cost proxy fusion 低成本代理融合**  
  EN: Kennedy–O’Hagan **autoregressive co-kriging** (ρ·f_L + δGP)  
  CN: Kennedy–O’Hagan 自回归 Co-Kriging（ρ·f_L + δGP）

- **Active sampling 主动采样**  
  EN: Mutual-information (logdet) greedy + **minimum-distance** constraint  
  CN: 互信息 logdet 贪心 + **最小距离**约束

- **Evaluation 评估完整**  
  EN: 10-fold CV (LOO-like) for MAE/RMSE/CRPS + hyper-parameter sweep heatmaps  
  CN: 10-fold 近似 LOO 的 MAE/RMSE/CRPS；超参扫描热力图

---

## 🗂 Project Tree / 目录结构

```

physics-informed-kriging-pro/
├─ requirements.txt                 # 固定依赖 / pinned deps
├─ src/
│  ├─ **init**.py
│  ├─ utils/
│  │  ├─ **init**.py
│  │  └─ geo.py                    # 坐标旋转：x,y ↔ 沿流/横流
│  └─ models/
│     ├─ **init**.py
│     ├─ nonstationary\_kernels.py  # 非平稳(Gibbs风格)旋转核
│     ├─ physics\_background.py     # PDE 背景均值（对流–扩散）
│     └─ advanced\_gp.py            # GP 封装（核+均值+屏障）
├─ scripts/
│  ├─ generate\_synth.py            # 合成数据 + 低价代理
│  ├─ run\_baseline.py              # 基线（图+指标+预测表）
│  ├─ run\_cokriging.py             # Co-Kriging 融合
│  ├─ select\_next\_samples.py       # 主动采样点选择
│  └─ sweep\_lengths.py             # (lp, lc) 扫描 → 热力图
├─ data/
│  ├─ synth\_points.csv
│  ├─ grid\_coords.csv
│  ├─ flow\_meta.json
│  ├─ proxy\_grid.csv
│  ├─ proxy\_points.csv
│  └─ malaysia\_coast\_example.geojson  # 占位海岸线（陆地多边形）
├─ figures/.gitkeep
└─ summary/.gitkeep

````

> ✅ 仓库自带 `data/` 示例，可直接运行；无需先生成数据。  
> Data is pre-baked; you can run straight away.

---

## 🧰 Environment / 环境

**Recommended**: Python 3.10–3.11（Windows 轮子更稳定）  
Recommended Python 3.10–3.11 for stable wheels on Windows.

### Windows (PowerShell)
```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
# 若遇“脚本被禁用”，管理员 PowerShell 执行：
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -r requirements.txt
````

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 若使用 GPU，请按你的 CUDA 版本安装对应的 PyTorch；CPU 直接用官方 CPU wheel 即可。

---

## 🚀 Quickstart / 开始

### A) Baseline（PDE 背景 + 屏障核）

```bash
python scripts/run_baseline.py \
  --use_pde_background \
  --barrier_geojson data/malaysia_coast_example.geojson
```

**Artifacts 产物**

* `figures/mean_map.png` — Posterior mean / 后验均值
* `figures/std_map.png` — Uncertainty / 后验标准差
* `figures/metrics.json` — MAE/RMSE/CRPS（10-fold）
* `data/grid_pred.csv` — 全网格均值与不确定性

---

## 🔁 Synthetic Data & Proxy / 合成数据与代理

重新生成（可调样本数、网格、流速、噪声等）：

```bash
python scripts/generate_synth.py \
  --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42
```

同时会输出：

* `proxy_grid.csv`、`proxy_points.csv`（模拟“遥感浊度指数”等低成本代理）
  → 用于 **Co-Kriging** 融合。

---

## 🗺 Real Coastline Replacement / 真实海岸线替换

将海岸线 GeoJSON 放到 `data/`（**多边形代表陆地**），运行时指定：

```bash
python scripts/run_baseline.py \
  --use_pde_background \
  --barrier_geojson data/your_malaysia_coast.geojson
```

屏障核会检测任意两点连线是否穿越陆地；穿越次数越多，相关性按指数衰减，避免跨陆“穿透预测”。

---

## 🧭 Active Sampling / 主动采样

互信息 logdet 贪心 + **最小距离约束**，输出下一轮采样点：

```bash
python scripts/select_next_samples.py --k_next 8 --min_dist 3.0
```

输出：`figures/next_points.csv`（建议坐标 & 当前不确定性）

---

## 🤝 Co-Kriging (Low-cost Proxy Fusion) / 低价代理融合

Kennedy–O’Hagan 自回归：`z(x) = ρ·f_L(x) + δ(x)`，其中 `f_L` 为低价代理拟合

```bash
python scripts/run_cokriging.py --n_lowfit 800 --lf_length 20 --hf_length 15
```

输出：

* `figures/mean_cok.png`, `figures/std_cok.png`
* `data/grid_pred_cok.csv`

---

## 🔥 Hyper-parameter Sweep / 参数扫描与热力图

对流向/横向长度尺度 `(lp, lc)` 网格扫描，生成 RMSE 热力图：

```bash
python scripts/sweep_lengths.py \
  --lp_list 20 30 40 --lc_list 6 8 12 --use_pde_background
```

输出：

* `summary/all_metrics.csv`
* `summary/heatmap.png`

---

## 🧷 Common Flags / 常用参数

`run_baseline.py`

* `--use_pde_background`：启用 PDE 背景均值（否则为沿流/横流二次多项式）
* `--kappa`、`--c_in`、`--source_amp`、`--source_x`、`--source_y`：扩散系数、入流浓度、源项
* `--barrier_geojson`、`--barrier_gamma`：屏障核与穿越惩罚强度
* `--length_parallel`、`--length_cross`、`--rq_alpha`：核长度与 RQ 形状参数
* `--nonstat_boost_along`、`--nonstat_boost_cross`、`--nonstat_tau`：非平稳长度随“离岸距离”的增强幅度与尺度
* `--no_opt`、`--n_restarts`：关闭/开启超参优化与重启次数

---

## 🧪 Reproduce in One Go / 复现实验

```bash
# 1) 生成合成数据 + 代理
python scripts/generate_synth.py --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42

# 2) 终极基线（PDE + 屏障 + 非平稳核）
python scripts/run_baseline.py --use_pde_background --barrier_geojson data/malaysia_coast_example.geojson

# 3) 主动采样建议
python scripts/select_next_samples.py --k_next 8

# 4) 低价代理融合
python scripts/run_cokriging.py

# 5) 超参扫描与热力图
python scripts/sweep_lengths.py --lp_list 20 30 40 --lc_list 6 8 12 --use_pde_background
```


---

## 🧱 Tips & Troubleshooting / 常见问题

* **Windows 轮子安装失败 / 安装慢**：建议 Python 3.10–3.11；必要时切换国内源或使用 `pip --default-timeout 100`。
* **地理屏障不生效**：检查 GeoJSON 里陆地是否为**闭合多边形**，坐标是否与数据同一坐标系（示例为归一化网格坐标或经纬度统一）。
* **主动采样点过密**：提高 `--min_dist`，或在已有点集周围自动屏蔽邻域。
* **数值爆炸/收敛慢**：适度上限/下限核长度与噪声先验，或先关闭优化用网格搜索热启动。

---

## 📜 Citation / 致谢

If you use this repository, please cite the PIK/kriging literature (advection–diffusion physics-informed GP, co-kriging) and acknowledge this implementation.
若在科研中使用本仓库，请引用相关 PIK/Co-Kriging 文献，并标注本仓库为实现参考。

