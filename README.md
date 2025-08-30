# Physics-Informed Kriging (PIK)

> 高可信空间插值与“下一步采样”解决方案（面向水质热点等环境场景）
> High-confidence spatial interpolation & next-best sampling for environmental hotspots

---

## 目录 / Table of Contents

* [亮点 / Highlights](#亮点--highlights)
* [为何选择 PIK？/ Why PIK?](#为何选择-pik--why-pik)
* [项目结构 / Project Structure](#项目结构--project-structure)
* [环境与安装 / Environment & Installation](#环境与安装--environment--installation)
* [快速上手 / Quickstart](#快速上手--quickstart)
* [一键复现实验 / Reproduce in One Go](#一键复现实验--reproduce-in-one-go)
* [数据与代理 / Data & Proxy](#数据与代理--data--proxy)
* [常用参数 / Common Flags](#常用参数--common-flags)
* [扩展示例 / Extended Demos](#扩展示例--extended-demos)
* [结果产物 / Outputs](#结果产物--outputs)
* [排错小抄 / Troubleshooting](#排错小抄--troubleshooting)
* [引用与致谢 / Citation & Acknowledgements](#引用与致谢--citation--acknowledgements)

---

## 亮点 / Highlights

* 物理先验（非稳态 PDE）：求解时变对流–扩散方程，得到随时间演化的背景均值 m(x,y,t)，作为 GP 的时间依赖背景；并通过线性/仿射校准提升泛化。
* Kernel Deep Learning（DKL）：用神经网络作为特征提取器，将 \[x,y,t,aux] 映射到嵌入空间，再在嵌入上做 GP（RBF/Matérn/RQ + Scale/ARD + 屏障核），兼顾表达力与不确定性量化。
* 丰富核函数组合：RBF/Matérn（沿流/横流 ARD） + RationalQuadratic（多尺度） + 非平稳（Gibbs-like）× 海岸屏障核 + 白噪声；支持时空核与输入依赖长度尺度。
* 低成本代理融合：基于 Kennedy–O’Hagan 自回归 Co-Kriging，以少量高保真 + 海量低保真实现性价比最优的预测。
* 主动采样（Next-Best Sampling）：基于互信息（A-opt/logdet）贪心策略，配合最小距离约束，给出下一步采样坐标建议。
* 严格评估：近 LOO 的 10-fold CV，报告 MAE / RMSE / CRPS / PIT；附超参扫描与热力图辅助诊断。

---

## 为何选择 PIK？ / Why PIK?

环境监测（如水质、空气污染）常呈强时空耦合：污染羽团随时间扩散/输运，海岸线/地形构成天然屏障，且观测稀疏。传统平稳/静态 Kriging 假设因此失效。本项目将非稳态物理（对流–扩散 PDE，随时间演化）与概率建模（非平稳核 + DKL + Co-Kriging）结合，兼顾可解释性、鲁棒性与采样效率。

---

## 多保真 + 主动采样 / Multi‑Fidelity + Active Sampling

**模块目标**：在 Physics‑Informed Kriging baseline 基础上，引入**多保真 Co‑Kriging（低保真代理 + 少量高保真观测）**与**主动采样（variance greedy / A‑optimal）**，在稀疏观测条件下显著提升预测精度，并给出下一轮最优采样点。

**多保真建模（Kennedy‑O'Hagan）**：

* 低保真 y\_L(x,t) \~ GP(m\_L, k\_L)
* 残差 δ(x,t) \~ GP(0, k\_δ)
* 高保真 y\_H(x,t) = ρ · y\_L(x,t) + δ(x,t) + ε\_H
  其中 ρ 为尺度/相关系数，k\_L 与 k\_δ 可用各向异性 + 非平稳 + 屏障核的和/积组合。

**主动采样策略**：

* Variance greedy：在候选集合 argmax σ²(x,t)。
* A‑optimal（A‑opt）：贪心近似最小化后验协方差迹（等价最大化信息增益）。支持批量选点与**最小距离约束**避免过密。

**使用流程**：

```bash
# 1) 训练多保真 Co-Kriging（先拟合低保真，再融合高保真）
python scripts/run_cokriging.py \
  --n_lowfit 800 --lf_length 20 8 --hf_length 15 6

# 2) 选择下一轮采样点（A-opt 或方差贪心）
python scripts/select_next_samples.py --k_next 8 --strategy aopt --min_dist 3.0

# 3) 采集新高保真样本后，增量更新
python scripts/run_cokriging.py --warm_start
```

**产物**：

* `figures/mean_cok.png`, `figures/std_cok.png`（多保真融合效果）
* `figures/acq_map.png`（采样效用/方差热图）
* `figures/next_points.csv`、`figures/next_points_ranked.csv`（建议点与得分）
* `results/mf_active_history.json`（每一轮的指标与参数追踪）

**实践建议**：

* 对代理与实测做**尺度/偏置校准**（ρ 的初值与边界）；必要时对 y 进行标准化。
* 若候选点靠岸聚集，调大 `--min_dist` 或在陆地内设置屏蔽；A‑opt 通常较 variance greedy 更稳健。
* 预算受限时，可设置**成本权重**（低保真成本远低于高保真）以平衡探索‑开销。

---

## 项目结构 / Project Structure

```text
physics-informed-kriging-pro/
├─ .github/workflows/            # CI 配置（如格式检查、单元测试）
├─ data/                         # 示例数据与地理边界（可直接运行）
├─ figures/                      # 可视化输出（均值/不确定性/热力图等）
├─ results/                      # 评估指标、追溯信息
├─ scripts/                      # 一键脚本（基线、Co-Kriging、主动采样、扫描等）
├─ src/                          # 核心实现（物理背景、核函数、GP 封装等）
├─ summary/                      # 汇总表与图（如 all_metrics.csv, heatmap.png）
├─ tests/                        # 单元/回归测试
├─ README.md
└─ requirements.txt
```

> 说明：仓库自带 `data/` 示例，可**开箱即跑**。

---

## 环境与安装 / Environment & Installation

**推荐 Python 3.10–3.11**（若使用 GPU，按你的 CUDA 版本安装对应 PyTorch；仅 CPU 则安装官方 CPU wheels）。

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 如遇“脚本被禁用”，在管理员 PowerShell 执行：
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -r requirements.txt
# 如需（CPU 示例）：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install gpytorch
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 如需：pip install torch gpytorch
```

---

## 快速上手 / Quickstart

### A) 时空基线（非稳态 PDE 背景 + 屏障核）

> 时间维度以列 `t` 表示；若仅有静态数据，可将 `t=0` 作为单一时间片。

```bash
python scripts/run_baseline.py \
  --use_pde_background \
  --barrier_geojson data/malaysia_coast_example.geojson
```

**产物**

* `figures/mean_map_*.png` — 不同时间片的后验均值
* `figures/std_map_*.png` — 不确定性时间序列
* `results/metrics.json` — 10-fold MAE / RMSE / CRPS / PIT
* `data/grid_pred_t.csv` — 全网格随时间的均值与不确定性

---

### B) Kernel Deep Learning（DKL）

> 使用神经网络作为特征提取器，再在嵌入上做 GP（需 `torch` + `gpytorch`）。

```python
# 简要示例：将 [x,y,t,aux] → φ_θ(x)
import torch, torch.nn as nn, gpytorch

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim=3, hid=64, emb=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, emb)
        )
    def forward(self, x):
        return self.net(x)

# 在 GP kernel 中使用嵌入（与屏障核/各向异性核可做和/积组合）
feature_extractor = FeatureExtractor(in_dim=3)  # 例如 [x,y,t]
base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=32)
emb_kernel  = gpytorch.kernels.ScaleKernel(base_kernel)
```

**产物**

* `figures/mean_dkl_*.png`, `figures/std_dkl_*.png`
* `data/grid_pred_dkl.csv`

---

### C) 多保真 Co-Kriging（融合低成本代理）

```bash
python scripts/run_cokriging.py \
  --n_lowfit 800 --lf_length 20 8 --hf_length 15 6
```

**产物**

* `figures/mean_cok.png`, `figures/std_cok.png`
* `data/grid_pred_cok.csv`

---

### D) 主动采样（互信息贪心 + 距离约束）

```bash
python scripts/select_next_samples.py --k_next 8 --strategy aopt --min_dist 3.0
```

**产物**

* `figures/next_points.csv` — 建议坐标与当前不确定性

---

### E) 多保真 + 主动采样循环（建议）

按 **C → D → 新采样 → C** 的顺序迭代若干轮；每轮将新高保真点并入训练集后 `--warm_start` 继续优化，并记录 `results/mf_active_history.json`。

## 一键复现实验 / Reproduce in One Go / Reproduce in One Go

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

## 数据与代理 / Data & Proxy

* 空间边界：`data/malaysia_coast_example.geojson`（用于屏障核与海岸穿越惩罚）。
* 观测/代理数据格式：

  * 点观测：`obs_points.csv` 至少包含 `x,y,z[,t]`；
  * 低保真代理：`proxy_points.csv` 至少包含 `x,y,z[,t]`；
  * 网格：`grid.csv` 至少包含 `x,y[,t]`；如含 `i,j` 便于还原矩阵热图。
* 时间维度：建议离散等间隔 `t`；若不等间隔，将 `t` 直接作为输入或做时间重采样。
* 合成数据生成：

  ```bash
  python scripts/generate_synth.py --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42
  ```

---

## 常用参数 / Common Flags（示例）

* PDE 背景：`--use_pde_background`, `--kappa`, `--c_in`, `--lambda`，`--source_amp/source_pos` 等；
* 屏障核：`--barrier_geojson`, `--barrier_gamma`（穿越惩罚强度）。
* DKL：`--use_dkl`, `--backbone {mlp,resnet}`, `--emb_dim`，`--freeze_backbone`；
* 时空核：`--length_parallel/--length_cross`（ARD），`--rq_alpha`（多尺度），`--time_length`（时间长度尺度），`--nonstat_tau`（非平稳尺度）。
* 优化：`--n_restarts`, `--lr`, `--max_iter`；
* 主动采样：`--k_next`, `--strategy {var,aopt}`, `--min_dist`。

> 小贴士：先用扫参脚本获取合理范围，再用 DKL+非稳态核做局部精调，可显著提升稳定性与精度。

---

## 扩展示例 / Extended Demos

```bash
# 1) 非稳态羽团（PDE 背景）
python scripts/run_dynamic_plume.py

# 2) DKL 对比（与纯核法）
python scripts/run_dkl_highdim.py

# 3) 非稳态羽团 + DKL 综合示例
python scripts/run_dynamic_dkl.py
```

**示例产物**

* `figures/plume_posterior_t.png`（时空均值）
* `figures/dkl_nll.png`, `figures/dkl_rmse.png`
* `data/dynamic_dkl_pred.pt`

> 说明：如脚本名与本地不同，请以仓库实际目录为准调整 `import` 与命令。

---

## 结果产物 / Outputs

* 时空地图：`figures/mean_map_*.png`, `figures/std_map_*.png`（多时间片）。
* 评估：`results/metrics.json`、`summary/all_metrics.csv`（MAE / RMSE / CRPS / PIT 与交叉验证记录）。
* 建议点：`figures/next_points.csv`（主动采样）。
* 网格预测：`data/grid_pred*.csv`（含 Co-Kriging / DKL 版本）。
* 热力图：`summary/heatmap.png`；可选导出 `gif/mp4` 时间序列。

---

## 排错 / Troubleshooting

* ModuleNotFoundError（包路径）：将 `pik_ext.*` 与 `src.models.*` 的 `import` 与仓库实际结构对齐。
* AttributeError: cannot assign module before `Module.__init__()`：在自定义 `nn.Module`/GP 模型中，务必先调用 `super().__init__()` 再注册子模块。
* 海岸屏障无效：确认 GeoJSON 为闭合多边形，坐标系/尺度与数据一致；必要时做归一化。
* 主动采样点过密：增大 `--min_dist` 或在已有点邻域设屏蔽。
* 数值不稳（非稳态 PDE）：减小时间步长 Δt、采用隐式/半隐式格式；为长度尺度/噪声设置合理先验/边界；先网格扫参热启动再优化。
* DKL 过拟合：

  * 对嵌入使用 `weight_decay`/`dropout`；
  * 先冻结骨干再联合微调；
  * 使用 LOVE 近似或降低嵌入维度；
  * 开启早停并监控 `PIT`/`CRPS`。
* Windows 安装慢/失败：优先 Python 3.10–3.11；必要时切换国内源，或 `pip --default-timeout 100`。

---

## 引用与致谢 / Citation & Acknowledgements

如在科研/生产中使用本仓库，请引用**物理信息 Kriging/GP**、**对流–扩散先验**与 **Co-Kriging** 相关文献，并注明本实现为参考。
If you use this repository, please cite the key literature on physics‑informed kriging/GP (advection–diffusion priors, co‑kriging) and acknowledge this implementation.

---

### 贡献 / Contributing

欢迎提交 Issue/PR 修复 Bug、补充数据与示例、或扩展核函数/主动采样策略。建议配套最小可复现实验与单测。

### 维护者 / Maintainer

本仓库由原作者维护。若有问题与建议，请在 Issues 中反馈。
