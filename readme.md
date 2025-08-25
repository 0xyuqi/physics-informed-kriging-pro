## README

```markdown
多保真 Co-Kriging + 主动采样

本模块基于 yuqi 同学的 physics-informed Kriging baseline，扩展了多保真建模与主动采样策略，进一步提升预测精度与实际应用价值。


目录结构


src/
models/cokriging.py      # Co-Kriging 实现
design/acquisition.py    # 主动采样策略
scripts/
run\_cokriging.py         # 跑多保真预测并输出结果
select\_next\_samples.py   # 主动采样点选择脚本

````

---

## 快速开始

### 1. 生成数据
先用仓库自带的脚本生成合成数据：
```bash
python scripts/generate_synth.py --n_obs 40 --grid 80 --noise 0.1 --vx 1.0 --vy 0.3 --seed 42
````

### 2. 跑多保真 Co-Kriging

```bash
python scripts/run_cokriging.py --n_restarts 2 --lf_length 20 8 --hf_length 15 6
```

输出：

* `figures/mean_cok.png`（预测均值）
* `figures/std_cok.png`（预测方差）
* `data/grid_pred_cok.csv`（网格预测结果）

### 3. 主动采样（选择下一轮测点）

**方差贪心：**

```bash
python scripts/select_next_samples.py --k_next 8 --strategy var --min_dist 3.0
```

**A-optimal 近似互信息：**

```bash
python scripts/select_next_samples.py --k_next 8 --strategy aopt --min_dist 3.0
```

输出：

* `figures/next_points.csv`（下一轮采样点）

---

## 环境依赖

* Python 3.9+
* numpy, pandas, matplotlib
* scikit-learn
* scipy

---

## 接口约定

* 输入：`data/proxy_points.csv`（低保真点），`data/obs_points.csv`（高保真点），`data/grid.csv`（候选网格）
* 输出：预测结果 CSV & 图片（figures/ 下）

---
