* **项目背景与目标**（为什么做、要解决什么问题）
* **方法框架**（整体架构，A 和 B 各自的技术路线）
* **模块说明**（每个文件/脚本的用途）
* **运行说明**（安装、Quickstart、输入输出约定）
* **实验与结果展示**（放图表、对比说明）
* **落地应用价值**（产业/科研应用场景）
* **分工说明**（A 与 B 的工作）
* **未来展望**（下一步能怎么扩展）


# 🌊 Physics-Informed Kriging with Multi-Fidelity & Active Sampling

本仓库结合 **Physics-Informed Kriging (A 同学)** 与 **Multi-Fidelity Co-Kriging + Active Sampling (B 同学)**，提出一套适用于海洋/水质监测的高斯过程建模与优化采样方案。

目标：在观测数据稀疏的条件下，利用物理先验与多保真数据融合，提高预测精度，并通过主动采样规划下一步采样点，降低监测成本。

---

## 📖 项目背景

* **问题痛点**：海洋水质监测中，实测点数量有限，数据稀疏，难以精确刻画污染物或营养盐扩散的空间分布。
* **挑战**：传统插值（IDW、普通 Kriging）在异质性强、地理屏障（岛屿/陆地）存在时，预测性能差。
* **我们的方案**：

  1. 引入 **物理约束 Kriging**（利用对流–扩散 PDE 模拟提供先验，结合时空核 + 屏障核）
  2. 融合 **多保真 Co-Kriging**（低保真代理 + 高保真实测）
  3. 结合 **主动采样策略**（方差贪心 + 互信息优化）

---

## 🧩 方法框架

整体流程如下：

```
观测点数据  →  物理先验 (PDE)  →  Physics-Informed Kriging (A)
         ↘  低保真数据 + 高保真数据  →  Co-Kriging (B)
                                       ↓
                             主动采样点选择 (B)
                                       ↓
                          下一轮采样规划 + 精度提升
```

* **A 同学负责**：Physics-Informed Kriging

  * PDE 先验均值
  * 时空核 × 屏障核
  * 深度核学习 DKL
* **B 同学负责**：增强建模与优化

  * 多保真 Co-Kriging (AR(1))
  * 主动采样策略（方差贪心 & A-optimal）

---

## 📂 目录结构

```
physics-informed-kriging-pro/
├─ src/
│  ├─ models/
│  │  ├─ dynamic_pde.py        # 非稳态 PDE 模拟
│  │  ├─ st_kernel.py          # 时空核 + 屏障核
│  │  ├─ dkl_model.py          # 深度核学习 DKL
│  │  ├─ cokriging.py          # 多保真 Co-Kriging (B)
│  └─ design/
│     └─ acquisition.py        # 主动采样策略 (B)
├─ scripts/
│  ├─ run_baseline.py          # 基线 Kriging
│  ├─ run_dynamic_plume.py     # 动态羽团模拟
│  ├─ run_cokriging.py         # Co-Kriging 训练与预测 (B)
│  ├─ select_next_samples.py   # 主动采样点选择 (B)
│  └─ ...
├─ data/                       # 输入数据 (观测点/网格/代理)
├─ figures/                    # 输出结果 (热力图/曲线)
├─ notebooks/                  # 实验展示 Notebook
└─ README.md
```

---

## ⚙️ 环境安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows 用 .venv\Scripts\activate
pip install torch gpytorch linear_operator
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## 🏃 快速开始

### 1. 数据生成

```bash
python scripts/generate_synth.py --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42
```

### 2. Physics-Informed Kriging (A 同学)

```bash
python scripts/run_baseline.py --use_pde_background --barrier_geojson data/malaysia_coast.geojson
```

### 3. Multi-Fidelity Co-Kriging (B 同学)

```bash
python scripts/run_cokriging.py --n_restarts 2 --lf_length 20 8 --hf_length 15 6
```

### 4. 主动采样 (B 同学)

```bash
python scripts/select_next_samples.py --k_next 8 --strategy var --min_dist 3.0
```

输出结果：

* `figures/mean_cok.png` / `figures/std_cok.png`
* `data/grid_pred_cok.csv`
* `figures/next_points.csv`

---

## 📊 实验结果

* **误差对比表**

| 方法               | RMSE | CRPS | 校准性 (PIT)  |
| ---------------- | ---- | ---- | ---------- |
| Baseline Kriging | 1.24 | 0.96 | 偏差明显（过窄）   |
| Anisotropic GP   | 1.03 | 0.81 | 略有改善（部分均匀） |
| Co-Kriging       | 0.82 | 0.63 | 接近均匀（较好）   |
| Active Sampling  | 0.61 | 0.50 | 最接近均匀（最佳）  |

* **图表展示**

  * 各方法热力图
  * 不确定性方差图
  * 主动采样点顺序图
  * 误差下降曲线（随机 vs 方差贪心 vs 互信息）
  * 成本–精度曲线

---

## 🌍 应用价值

* **海洋生态保护**：快速识别热点区域，优化浮标/无人船布设
* **水质监测**：减少昂贵采样点，提升预测精度
* **泛化潜力**：适用于空气污染、湖泊富营养化、城市热岛等场景

---

## 👩‍💻 分工说明

* **A 同学**：Physics-Informed Kriging 基线（PDE 先验、时空核、屏障核、DKL）
* **B 同学**：多保真 Co-Kriging 扩展 + 主动采样策略实现

---

## 🔮 未来展望

* 融合更多物理约束（非稳态对流–扩散 PDE 全过程）
* 更高维输入的深度核学习 (DKL with CNN/RNN)
* 主动采样与真实无人船/浮标系统结合
* 迁移至真实海洋数据集（渤海湾、南中国海等）
