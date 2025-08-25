# Physics-Informed Kriging (PIK-Dyn)

**非稳态 PDE + 深度核学习（DKL）+ 时空核 × 屏障核 + 主动采样 + Co-Kriging**

> 一套用于水质/污染羽团等场的**物理约束高斯过程**解决方案：
> 既可以“用物理减弱数据稀疏的歧义”，又能通过 **DKL** 适配高维复杂输入，还能主动规划下一步采样以快速降低不确定性。

---

##   Highlights | 亮点

* **非稳态 PDE 先验（动态羽团）**：用对流–扩散方程的数值模拟/谱法生成 $u(x,y,t)$，作为 GP 的**先验均值**，并提供 $\alpha,\beta$ 线性校准。
  *Unsteady advection–diffusion plume simulation used as GP mean; learnable affine calibration.*

* **深度核学习 DKL**：MLP 特征提取 + ExactGP，提升高维/非线性结构下的泛化与不确定性质量。
  *Deep feature extractor + GP improves fit in high-dimensional settings.*

* **时空核 × 屏障核**：空间核 $\times$ 时间核 的可分离结构；对跨陆连线进行**屏障衰减**，抑制“穿透式”错误相关。
  *Separable space–time kernel with barrier attenuation to prevent land-crossing correlations.*

* **低成本代理融合（Co-Kriging）**：自回归框架 $z_H = \rho\,f_L + \delta$，把遥感/廉价传感的低保真信息与少量高保真“真值”融合。
  *Autoregressive co-kriging fuses low-/high-fidelity sources.*

* **主动采样**：基于方差/互信息的贪心 + 最小距离约束，给出**下一轮采样点**，以最大化不确定性下降。
  *Greedy MI/variance with min-distance to plan next measurements.*

---

##   Repository Structure | 目录结构

```
physics-informed-kriging-pro/
├─ src/
│  ├─ models/
│  │  ├─ dynamic_pde.py        # 非稳态对流–扩散谱法模拟（T×H×W）
│  │  ├─ pde_mean.py           # PDE 体数据 → GP 先验均值（3D 双/三线性插值）
│  │  ├─ st_kernel.py          # 可分离时空核 + 屏障衰减
│  │  ├─ barrier.py            # 屏障核：Shapely/栅格 fallback
│  │  ├─ exactgp_st.py         # 时空 ExactGP 封装
│  │  ├─ dkl_model.py          # DKL（MLP 特征 + ExactGP，已修正 init 顺序）
│  │  └─ cokriging.py          # 自回归 Co-Kriging（两阶段）
│  └─ utils/
│     ├─ geo.py                # GeoJSON 读取/线段穿越检测工具
│     └─ sampling.py           # 主动采样（EPV/MI 贪心 + 距离约束）
├─ scripts/
│  ├─ run_baseline.py          # PDE 均值 + 时空核 + 屏障 → 基线图/指标
│  ├─ run_dynamic_plume.py     # 非稳态羽团 + 时空 GP（出 t=0.6 切片图）
│  ├─ run_dkl_highdim.py       # Plain GP vs DKL 基准（NLL/RMSE 柱状）
│  ├─ run_dynamic_dkl.py       # 动态 + DKL 综合（导出 .pt / 对比图）
│  ├─ run_cokriging.py         # 低/高保真融合演示（均值/方差图）
│  ├─ select_next_samples.py   # 主动采样点 CSV（k、最小距离）
│  └─ sweep_lengths.py         # (lp, lc) 长度尺度扫描 → 热力图
├─ data/                       # 示例/复现实验数据（可被脚本覆盖）
├─ figures/                    # 结果图件（脚本生成）
├─ results/                    # 指标 JSON 等
├─ summary/                    # 扫描统计与热力图
├─ tests/smoketest.py          # 冒烟测试（导入路径/版本兼容）
└─ .github/workflows/ci.yml    # 最小 CI（CPU 轮子，跑冒烟/基线）
```


##   Installation | 环境安装

> **Python**：建议 3.10–3.11（Windows 轮子更稳定）
> **PyTorch**：先装（按 CPU/GPU 选择命令），再装 GPyTorch 与其余依赖。

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 先装 PyTorch（示例为 CPU 版；若有 CUDA 请用官网命令替换）
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu
# 再装 GPyTorch 与其它依赖
pip install gpytorch==1.11 linear_operator==0.5.2
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/cpu   # 或按你CUDA版本
pip3 install gpytorch==1.11 linear_operator==0.5.2
pip3 install -r requirements.txt
```

---

##   Quickstart | 快速上手

> 仓库已带 `data/` 与示例脚本，直接运行即可生成图件到 `figures/`。

**1) 基线（PDE 均值 + 屏障核 + 时空核）**

```bash
python scripts/run_baseline.py --use_pde_background --barrier_geojson data/malaysia_coast_example.geojson
```

**产物**：`figures/mean_map.png`, `figures/std_map.png`, `results/metrics.json`

**2) 动态羽团（非稳态 PDE + 时空 GP）**

```bash
python scripts/run_dynamic_plume.py
```

**产物**：`figures/plume_posterior.png`（t=0.6 切片的均值/方差）

**3) 高维对比（Plain GP vs DKL）**

```bash
python scripts/run_dkl_highdim.py
```

**产物**：`figures/dkl_nll.png`, `figures/dkl_rmse.png`

**4) 动态 + DKL 综合**

```bash
python scripts/run_dynamic_dkl.py
```

**5) Co-Kriging 低/高保真融合**

```bash
python scripts/run_cokriging.py
```

**6) 主动采样（互信息/方差贪心 + 距离约束）**

```bash
python scripts/select_next_samples.py --k_next 8 --min_dist 0.1
```

**产物**：`figures/next_points.csv`

**7) 参数扫描（lp, lc）→ 热力图**

```bash
python scripts/sweep_lengths.py --lp_list 20 30 40 --lc_list 6 8 12
```

**产物**：`summary/all_metrics.csv`, `summary/heatmap.png`

---

##   Key Flags | 关键参数

* `run_baseline.py`

  * `--use_pde_background`：启用 PDE 先验（否则使用多项式趋势）
  * `--barrier_geojson` & `--barrier_gamma`：屏障核与惩罚强度
  * `--kappa`：扩散系数（PDE 中）
* `select_next_samples.py`

  * `--k_next`：下一轮选择点数
  * `--min_dist`：点与点之间的最小距离（防止扎堆）

---

##   Method Notes | 方法说明

1. **PDE 先验均值**：把 $u(x,y,t)$ 体数据注册为 3D 体（`pde_mean.py`），对任意 $(x,y,t)$ 做双/三线性插值得到 $\mu(x,y,t)$，并通过 $\alpha,\beta$ 学习矫正模拟与观测的系统偏差。
2. **Separable ST Kernel**：$\;k(\mathbf{x},t;\mathbf{x}',t')=k_s(\mathbf{x},\mathbf{x}')\cdot k_t(t,t')\;$；空间核可用 ARD-RBF / Matern，时间核常用 Matern($\nu=0.5$ / RBF)。
3. **Barrier Attenuation**：若两点连线穿越陆地（GeoJSON 或栅格掩膜检测），在核矩阵上乘以 $\exp(-\gamma)$ 的衰减，避免跨陆“穿透”。
4. **DKL**：MLP 将输入映射到潜空间，再用 ExactGP；这在复杂/高维输入下通常能得到更低的 NLL/RMSE。
5. **Co-Kriging**：先拟合低保真 GP，再在高保真子集上估计 $\rho$，最后对残差拟合 GP，预测时叠加。
6. **Active Sampling**：以当前方差/互信息作为效用，采用贪心 + 最小距离约束选择候选点，快速降低总体不确定性。

---

##   Reproducibility & CI | 复现与持续集成

* 运行冒烟测试：

```bash
python tests/smoketest.py
```

* 仓库含最小 CI（GitHub Actions），会在 CPU 环境安装依赖并跑导入/基线脚本，保障可复现性。([GitHub][1])

---

##   Troubleshooting | 常见问题

* **`ModuleNotFoundError: gpytorch`**：先装 PyTorch，再装 `gpytorch==1.11` 与 `linear_operator==0.5.2`。
* **`cannot assign module before Module.__init__()`**：已在 `dkl_model.py` 修复（先 `super().__init__()` 再注册子模块）。
* **Shapely/GeoJSON**：若本机无 `shapely`，屏障退化为无屏障；推荐按 `requirements.txt` 安装保证功能完备。
* **GPU**：使用 GPU 时，请按 PyTorch 官网上与你 CUDA 版本匹配的命令安装。

---

##   Use Cases | 适用场景

* 水环境热点/污染羽团插值与预报、监测布点优化
* 城市空气/气味投诉热区、湖泊富营养化区域识别
* 任意**稀疏观测 + 物理可表述**的时空场建模与采样规划

---

##   Citation | 引用

> 如果你的研究或比赛使用了本仓库的方法/代码，请在报告/论文中注明 “Physics-Informed Kriging with Unsteady PDE Prior and Deep Kernel Learning (PIK-Dyn)”。



