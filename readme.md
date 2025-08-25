# Physics-Informed Kriging (PIK) for Water-Quality Hotspots

**物理约束高斯过程（Kriging/GP）与主动采样优化**

> 面向**水质热点识别**与**监测网络优化**：在**零散人工监测**与/或**低价代理**数据条件下，提供高可信的**空间/时空插值**与**下一轮采样点**建议。核心思想是把**对流–扩散 PDE**作为物理先验，结合**高斯过程**、**协同克里金（多保真融合）**与**基于信息增益的主动采样**，形成一套端到端流程。

---

##   功能特性（Highlights）

* **Physics-informed Kriging（PIK）**：将**对流–扩散 PDE**的数值解作为**GP 先验均值/约束**，并对偏差进行**线性校准**（β）。与纯数据核方法相比，能在小样本下输出更稳健的均值与不确定度。相关理论可参考 PhIK/CoPhIK 系列与 APIK 的主动学习思想。([engineering.lehigh.edu][1], [ACM Digital Library][2], [arXiv][3])
* **各向异性/非平稳核**：沿/横流 ARD-RBF + RQ（多尺度）+ 可选**非平稳调制**（如随岸距变化）与**屏障核**（海岸/障碍不可穿越），并叠加白噪声。
* **协同克里金（Co-Kriging，多保真）**：融合**低成本代理**（如遥感浊度指数、粗格点数值模式）与少量高保真“真值”（实验室检测），提升热点定位与不确定性收敛。参考 CoPhIK。([ACM Digital Library][2])
* **主动采样（Active Sampling）**：用\*\*互信息（MI/log-det）\*\*或 **EPV（Expected Posterior Variance）**贪心选点，带**最小间距约束**，为“本轮→下一轮”采样提供方案。可借鉴 APIK 思路（把物理点/观测点联合设计）。([arXiv][3])
* **评估与可视化**：支持 LOO/ K-fold、**H-步时序外推**；输出 MAE/RMSE、**CRPS**（概率预报质量）、**热点识别 P/R**（阈值法）与地图/剖面图/不确定度图等。

---
##  方法与公式（简要）
给定观测集 $\mathcal{D}=\{(\mathbf{s}_i,t_i),\, z_i\}_{i=1}^N$，在物理先验 $m_\theta(\mathbf{s},t)$ 下建模：

$$
\begin{aligned}
f(\mathbf{s},t) &\sim \mathcal{GP}\!\left(
m_\theta(\mathbf{s},t)\,\beta,\;
k\!\left((\mathbf{s},t),(\mathbf{s}',t')\right)
\right),\\[6pt]
z_i &= f(\mathbf{s}_i,t_i) + \varepsilon_i,\qquad 
\varepsilon_i \sim \mathcal{N}(0,\sigma_n^2).
\end{aligned}
$$

**物理先验（PDE）：对流–扩散**

$$
\partial_t c(\mathbf{s},t) + \mathbf{u}\!\cdot\!\nabla c(\mathbf{s},t)
= \kappa \nabla^2 c(\mathbf{s},t) + q(\mathbf{s},t),
\qquad \text{BC/IC 给定}.
$$

- 稳态：$\partial_t c = 0 \;\Rightarrow\; m_\theta(\mathbf{s})$  
- 非稳态：时间推进得到 $m_\theta(\mathbf{s},t)$（亦可用多次物理模拟构造/修正先验与残差）。

**核函数（示例）**

- 空间核：
  $$
  k_s(\mathbf{s},\mathbf{s}')
  = \mathrm{RBF}_{\parallel}\!\cdot\!\mathrm{RBF}_{\perp}
    + \mathrm{RQ} + \mathrm{NonstationaryMod}
  $$
  （可叠加屏障核）
- 时间核：$k_t(t,t')$ 取 RBF / Matern（可加日周期核）  
- 时空核：$k\big((\mathbf{s},t),(\mathbf{s}',t')\big)=k_s(\mathbf{s},\mathbf{s}')\,k_t(t,t')$（可分离），或设计**非分离**核以表达顺流“传播滞后”  
- 观测噪声：$\sigma_n^2$（可异方差）

**多保真（Co-Kriging，自回归式）**

$$
f_H(\cdot)=\rho\,f_L(\cdot)+\delta(\cdot),\qquad
\delta\sim \mathcal{GP}\!\left(0,\,k_\delta\right).
$$

其中 $f_L$ 为低价代理（如遥感），$f_H$ 为高保真“真值”。

**主动采样**

$$
\mathcal{Q}^\star
= \arg\max_{\substack{\mathcal{Q}\subset \mathcal{C}\\|\mathcal{Q}|=K}}
\bigg[
\log\det\!\big(K_{\mathcal{Q}\mid \mathcal{D}}\big)
\ \text{或}\
\sum_{x\in\mathcal{Q}} \mathrm{Var}_{\text{post}}(x)
\bigg],
$$

并施加最小间距/屏障约束；可扩展到时空候选 $(\mathbf{s},t)$。


---

##   数据规范（Data Schema）

最少需以下表格/文件）：

| 文件                      | 作用       | 必要列                                             |
| ----------------------- | -------- | ----------------------------------------------- |
| `data/observations.csv` | 观测点      | `x,y[,t],value[,sigma]`（经纬度或投影坐标；`t` 可为时间戳或归一化） |
| `data/grid.csv`         | 预测网格     | `x,y[,t]`（静态场可省 `t`；时变场给出目标时刻或滑窗）               |
| `data/barrier.geojson`  | 屏障       | 闭合多边形（海岸/陆地/屏障），坐标系与观测一致                        |
| `data/proxy.csv`        | 低价代理（可选） | 与观测同一坐标/时间基准的 `x,y[,t],proxy_value`             |
| `config.yaml`           | 运行配置     | 见下方示例                                           |

> **坐标与投影必须统一**；`barrier.geojson` 必须闭合且与数据 CRS 一致（常见错误来源之一）。

---

##   配置文件示例（`config.yaml`）

```yaml
# 数据
data:
  obs: data/observations.csv
  grid: data/grid.csv
  proxy: data/proxy.csv        
  barrier: data/barrier.geojson 

# 物理先验（PDE）
physics:
  use_background: true
  kappa: 0.8        # 扩散系数
  u: [0.6, 0.2]     # 水流速度 (vx, vy)
  source:           # 可选：点源/边界源
    amp: 1.0
    xy: [120.3, 31.1]

# 核与噪声
kernel:
  space:
    length_parallel: 30
    length_cross: 8
    rq_alpha: 0.8
    nonstationary: { enable: true, tau: 0.2 }
    barrier: { enable: true, gamma: 12.0 }
  time:
    enable: false   # 非稳态开启：true
    type: matern32
    length: 6.0
  noise:
    homoskedastic: true
    sigma: 0.15

# 训练/推断
train:
  optimize_hyperparams: true
  restarts: 5
  seed: 42

# 主动采样
active:
  enable: true
  k_next: 8
  min_dist: 3.0
  objective: "mi"   # mi | epv

# 评估
eval:
  cv: "kfold"       # loo | kfold
  folds: 10
  hotspot_threshold: 0.6
  forecast_h: 3     # H-step 外推（非稳态时）
```

---

##   快速上手（推荐流程）

1. **准备数据**：把现有监测数据与可用代理按“数据规范”整理到 `data/`；检查坐标与屏障一致。
2. **运行基线（PIK）**：启用 `physics.use_background=true`；输出**均值/不确定度**地图、**指标**（MAE/RMSE/CRPS）与**全网格预测 CSV**。
3. **开启 Co-Kriging**：提供 `data/proxy.csv` 并在配置中启用多保真；比较与基线的 RMSE/CRPS 改善幅度。([ACM Digital Library][2])
4. **主动采样**：设置 `active.k_next` 与 `min_dist`；得到**下一轮建议点**（CSV + 可视化）。([arXiv][3])
5. **超参扫描**（可选）：对沿/横流长度、噪声与非平稳幅度做网格扫描，输出**热力图**与**最佳区间**；记录配置与随机种子便于复现实验。
6. **（如需）非稳态**：在 `kernel.time.enable=true`，并提供带 `t` 的网格与观测；见“非稳态扩展”。

> 提示：任何脚本/入口都应支持 `--config config.yaml` 与 `--help`；建议所有产物规范化输出到 `figures/`、`results/`、`summary/`。

---

##   结果产物（Outputs）

* **图件**：`figures/mean_map.png`, `figures/std_map.png`, `figures/hotspot_prob.png`, `figures/next_samples.png`, `summary/heatmap.png`
* **表格**：`results/metrics.json`（MAE/RMSE/CRPS/K-fold）、`results/hotspot_pr.csv`、`data/grid_pred.csv`（网格后验均值/方差/分位数）、`results/next_points.csv`
* **追溯**：保存 `config.yaml` 副本、随机种子、Git commit、训练日志到 `results/run-YYYYMMDD-HHMM/`

---

##   评估与指标

* **交叉验证**：LOO 或 K-fold（默认 10-fold）；
* **H-步外推**（非稳态）：按时间切片做滚动或 block-CV；
* **误差统计**：MAE / RMSE / **CRPS**（概率预报）；
* **热点识别**：阈值 $\tau$ 下计算 **Precision / Recall**、**F1**，或输出 **PR-AUC**。

---

##    非稳态（时变）PDE 扩展：最小改动清单

1. **数据**：在观测与网格中新增 `t` 列；标准化或使用统一时间单位。
2. **物理均值**：实现显/隐式时间推进（如 Crank–Nicolson），得到 $m_\theta(\mathbf{s},t)$；在 GP 中保留**线性校准 β**（抵消系统偏差）。
3. **核**：启用 `k_t`（RBF/Matern/周期）；必要时使用**非分离时空核**处理顺流“传播滞后”。
4. **训练**：长序列用**滑窗/分块**与诱导点；
5. **主动采样**：候选扩展到 $(\mathbf{s},t)$，并加**时间间隔约束**；
6. **评估**：加入 **H-步外推**与按时间分层的 K-fold。

> 相关思路与物理约束 GP/主动学习文献相符（PhIK/CoPhIK/APIK）。

---

##    常见坑位与排查

* **屏障无效/穿岸**：`barrier.geojson` 非闭合或坐标系不一致 → 修正 CRS/闭合面。
* **核尺度失衡**：沿/横流长度差异过大导致数值不稳 → 对数域优化并设合理先验/边界。
* **优化不收敛**：启用多重重启（`restarts`），先粗扫再精调；必要时固定部分参数。
* **不确定性过低**：代理质量差但权重过高 → 在 Co-Kriging 中放松 $\rho$ 或增大残差核；做好**异方差**建模。([ACM Digital Library][2])
* **时变误配**：非分离核缺乏“传播滞后” → 用随 $\Delta t$ 缩放的空间尺度或引入流向旋转的时空核。

---

##   代码与脚手架（建议）

* 统一入口：`python -m pik.run --config config.yaml`（建议）

  * 子命令：`fit`（训练/插值）、`active`（选点）、`eval`（评估）、`sweep`（扫描）；
* 模块划分：

  * `pik/physics/`（PDE 解算 & 插值）、`pik/kernels/`（空间/时间/屏障/非稳态核）
  * `pik/models/`（GP/Co-Kriging 封装）、`pik/active/`（MI/EPV + 约束）
  * `pik/io/`（CSV/GeoJSON/配置）、`pik/viz/`（地图/剖面/热力图）
* 测试：`tests/` 覆盖核边界、屏障约束、物理均值校准与 MI 选点一致性。

> 如果你的仓库已经有对应脚本与目录，**保持现有命名即可**；README 的组织对“单入口/多脚本”两种结构都兼容。

---

##   贡献（Contributing）

* Issue/PR 欢迎：新增核类型、改进屏障核、添加**时空非分离核**、更多主动采样目标（如 BALD）、更全面的评估协议等。
* 提交前：请运行 `pre-commit` & `pytest -q`，并补充最小复现与基准对照。



##   参考文献（主要方法脉络）

1. **PhIK（Physics-Informed Kriging）**：Yang, Tartakovsky & Tartakovsky, *A Physics-Informed Gaussian Process Regression Method for Data-Model Convergence*, 2018. ([engineering.lehigh.edu][1])
2. **CoPhIK（Physics-Informed Co-Kriging，多保真）**：Yang et al., *Physics-informed CoKriging*, J. Comput. Phys., 2019. ([ACM Digital Library][2])
3. **APIK（Active PIK）**：Chen et al., *Active Physics-Informed Kriging (APIK)*, 2020. ([arXiv][3])

---

