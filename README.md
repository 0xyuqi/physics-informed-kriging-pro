
# Physics-Informed Kriging (PIK)

> 高可信空间插值与 **最优下一步采样**（水质热点等环境场景）  
> High-confidence spatial interpolation + **next-best sampling** for environmental hotspots

---

## ✨ Highlights / 亮点

- **Physics prior 物理先验**  
  Steady advection–diffusion PDE solved on grid ⇒ bilinear interpolation as GP mean with linear calibration.  
  稳态对流–扩散方程在网格上求解，**双线性插值**作为 GP **均值**并线性校准。

- **Kernels 核函数**  
  RBF (ARD along/cross-flow) + RationalQuadratic (multi-scale) + non-stationary Gibbs-like × coastline **barrier** + white noise.  
  RBF（ARD：流向/横向）+ RQ（多尺度）+ **非平稳** Gibbs 风格核 × **海岸屏障核** + 白噪声。

- **Low-cost proxy fusion 低成本代理融合**  
  Kennedy–O’Hagan **autoregressive co-kriging** (ρ·f_L + δGP).  
  Kennedy–O’Hagan 自回归 Co-Kriging（ρ·f_L + δGP）。

- **Active sampling 主动采样**  
  Mutual-information (logdet) greedy + **minimum-distance** constraint.  
  互信息 logdet 贪心 + **最小距离**约束。

- **Evaluation 评估**  
  10-fold CV (LOO-like) for MAE/RMSE/CRPS + hyper-parameter sweep heatmaps.  
  10-fold 近似 LOO 的 MAE/RMSE/CRPS；超参扫描热力图。

---

## 🗂 Project Tree / 目录结构

```

physics-informed-kriging-pro/
├─ .github/workflows/             
├─ data/                             # 示例数据与地理边界
├─ figures/                          # 出图目录
├─ pik\_ext/                          # 扩展实现（DKL、时空核等）
├─ results/                          # 指标/追溯信息
├─ scripts/                          # 可直接运行的脚本
├─ src/                              # 基础实现（物理均值、核、GP封装等）
├─ summary/                         
├─ tests/                            
├─ README.md
└─ requirements.txt

````

>  仓库自带 `data/` 示例，可直接运行（无需先生成数据）。  
> Data in `data/` allows **out-of-box** runs.

---

##  Environment / 环境

**Recommended / 推荐**：Python **3.10–3.11**
如用 GPU，请按你的 CUDA 版本安装对应 PyTorch；仅 CPU 则使用官方 CPU wheels。

### Windows (PowerShell)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
# 若遇“脚本被禁用”，以管理员 PowerShell 执行：
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -r requirements.txt
# 若脚本涉及 GPytorch/torch，可按需补装（CPU示例）：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install gpytorch
````

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 如需：pip install torch gpytorch
```

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

### B) Low-cost proxy fusion（Co-Kriging / 低价代理融合）

```bash
python scripts/run_cokriging.py \
  --n_lowfit 800 --lf_length 20 --hf_length 15
```

**Outputs**

* `figures/mean_cok.png`, `figures/std_cok.png`
* `data/grid_pred_cok.csv`

---

### C) Active sampling（互信息贪心 + 距离约束）

```bash
python scripts/select_next_samples.py --k_next 8 --min_dist 3.0
```

**Outputs**

* `figures/next_points.csv`（建议坐标与当前不确定性）

---

### D) Hyper-parameter sweep（参数扫描 → 热力图）

```bash
python scripts/sweep_lengths.py \
  --lp_list 20 30 40 --lc_list 6 8 12 --use_pde_background
```

**Outputs**

* `summary/all_metrics.csv`
* `summary/heatmap.png`

---

##  Synthetic Data & Proxy / 合成数据与代理

重新生成（可调样本数、网格、流速、噪声等）：

```bash
python scripts/generate_synth.py \
  --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42
```

同时输出：

* `proxy_grid.csv`、`proxy_points.csv`（模拟遥感等低成本代理；供 Co-Kriging 使用）

---

##  Extended Demos（DKL & Dynamic Plume）/ 扩展示例

> 若你已添加扩展模块（`pik_ext/` 或 `src/models/` 中的 DKL、时空核、动态羽团），可运行下列脚本（文件名按你仓库的脚本而定）：

```bash
# 动态羽团 + 时空 GP
python scripts/run_dynamic_plume.py

# DKL 高维基准对比
python scripts/run_dkl_highdim.py

# 动态羽团 + DKL 综合示例
python scripts/run_dynamic_dkl.py
```

**Artifacts 产物（示例）**

* `figures/plume_posterior.png`
* `figures/dkl_nll.png`, `figures/dkl_rmse.png`
* `data/dynamic_plume_fields.pt`, `data/dynamic_plume_obs.pt`, `data/dynamic_dkl_pred.pt`

> 注：如你的脚本中存在包路径差异（`pik_ext.*` vs `src.models.*`），请按实际路径调整 `import`。
> 若使用 GPytorch 模型，请优先使用 Python 3.10–3.11 以避免兼容性问题。

---

##  Common Flags / 常用参数（`run_baseline.py`）

* `--use_pde_background`：启用 PDE 背景均值（否则为沿流/横流二次多项式）
* `--kappa` `--c_in` `--source_amp` `--source_x` `--source_y`：扩散系数、入流浓度、源项
* `--barrier_geojson` `--barrier_gamma`：屏障核与穿越惩罚强度
* `--length_parallel` `--length_cross` `--rq_alpha`：核长度与 RQ 形状参数
* `--nonstat_boost_along` `--nonstat_boost_cross` `--nonstat_tau`：非平稳长度随“离岸距离”的增强幅度与尺度
* `--no_opt` `--n_restarts`：关闭/开启超参优化与重启次数

---

##  Reproduce in One Go / 复现实验（一步到位）

```bash
# 1) 合成数据 + 代理
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

##  Tips & Troubleshooting / 常见问题

* **ModuleNotFoundError（包路径）**：按仓库实际结构将 `pik_ext.*` 与 `src.models.*` 的 `import` 对齐。
* **AttributeError: cannot assign module before `Module.__init__()`**：在自定义 `nn.Module`/GP 模型 `__init__` 里需**先**调用 `super().__init__()` 再挂子模块。
* **海岸屏障无效**：确认 GeoJSON 中陆地为**闭合多边形**；坐标系与数据一致（示例为归一化或统一经纬）。
* **主动采样点过密**：增大 `--min_dist` 或在已有点邻域设屏蔽。
* **收敛慢/数值不稳**：为长度尺度、噪声设定合理先验与边界；或先网格搜索热启动再优化。
* **Windows 安装慢/失败**：优先 Python 3.10–3.11；必要时切换国内源，或 `pip --default-timeout 100`。

---

##  Citation / 致谢

If you use this repository, please cite the key literature on physics-informed kriging/GP (advection–diffusion priors, co-kriging) and acknowledge this implementation.
若在科研中使用本仓库，请引用相关 PIK / Co-Kriging 文献，并注明本实现为参考。
