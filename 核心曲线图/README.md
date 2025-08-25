
---

# 📊 实验结果解释

### 1. RMSE vs 采样轮数

* **中文解释**：随着采样轮数的增加，所有方法的均方根误差（RMSE）都逐步下降，说明增加观测点确实能提升模型精度。但不同采样策略的收敛速度差别显著：随机采样下降最慢；方差贪心策略集中在高不确定性区域，收敛更快；A-optimal 策略综合考虑全局信息，下降最快，最终 RMSE 最低。
* **English**: As the sampling rounds increase, the root mean square error (RMSE) decreases across all methods, indicating that adding more observations improves model accuracy. However, the convergence speed differs significantly: random sampling is the slowest, variance-greedy sampling converges faster by targeting high-uncertainty regions, and A-optimal (mutual information based) achieves the fastest and lowest RMSE.

---

### 2. CRPS vs 采样轮数

* **中文解释**：CRPS（连续秩概率得分）衡量的是预测分布与真实值的贴合度。曲线显示，主动采样策略不仅能降低均值预测误差，还能提升整个分布的质量。其中，A-optimal 策略在 10 轮后达到最低 CRPS，说明其不确定性估计最可靠。
* **English**: The Continuous Ranked Probability Score (CRPS) measures how well the predictive distribution matches the true values. The curves show that active sampling strategies improve not only mean prediction accuracy but also distribution quality. Among them, A-optimal achieves the lowest CRPS after 10 rounds, demonstrating the most reliable uncertainty estimation.

---

### 3. PIT 校准直方图

* **中文解释**：PIT 分布用于评估预测分布的校准性。理想情况是均匀分布：

  * Baseline Kriging 结果集中在中间，说明模型过于自信，低估了不确定性。
  * Anisotropic GP 有所改善，但仍存在偏差。
  * Co-Kriging 接近均匀，表明分布更合理。
  * Active Sampling 几乎完全均匀，说明模型的概率预测与真实分布高度一致，校准最佳。
* **English**: PIT histograms evaluate calibration of predictive distributions. Ideally, the values should follow a uniform distribution:

  * Baseline Kriging shows strong central bias, indicating overconfidence.
  * Anisotropic GP improves slightly but still shows deviation.
  * Co-Kriging is close to uniform, reflecting a more reasonable distribution.
  * Active Sampling is nearly uniform, meaning the predictive probabilities align very well with reality, showing the best calibration.

---

### 4. 结果总结

* **中文总结**：综合 RMSE、CRPS 与 PIT 三个指标，可以看到从 Baseline → Anisotropic → Co-Kriging → Active Sampling，模型的精度、分布质量和校准性都逐步提升。尤其是主动采样显著加速了误差下降，并提升了模型的可靠性。
* **English Summary**: Combining RMSE, CRPS, and PIT results, we observe a clear progression: Baseline → Anisotropic → Co-Kriging → Active Sampling. Accuracy, distribution quality, and calibration all improve step by step, with active sampling significantly accelerating error reduction and enhancing model reliability.

---
