# A-Multi-Seed-Reliability-Benchmark

**论文**：*TD-Target Discrepancy Does Not Predict Multi-UAV Policy Performance: A Multi-Seed Reliability Benchmark*
（中文译名：TD 目标偏差不能预测多无人机策略性能——一个多种子可靠性基准）
**作者**：Chen Li、Zikai Huang

本仓库是上述论文的**可复现材料**：多无人机协同路径规划场景下"Q 值估计偏差抑制算子"的**多种子可靠性基准**，包含训练与评估代码、当时实际运行的复现脚本，以及论文每张表、每张图背后的**原始测量数据**。

---

## 一、这项工作是什么

论文把两个**已发表**的 Q 值估计偏差抑制算子，实例化到一个小型多无人机协同路径规划测试台，并与两个标准多智能体基线、一个独立单机基线一起做**多种子**评估：

| 方法 | 说明 |
|---|---|
| **MADDPG** | 多智能体 DDPG 基线 |
| **MATD3** | 多智能体 TD3（clipped double Q）基线 |
| **MUAV-STD3** | 本文对 **softmax Bellman 算子**（MASTD3，见 [21]）的实例化 |
| **MUAV-MTD3** | 本文对**智能体间互估计**（M2ATD3，见 [21]）的实例化 |
| **TD3（独立）** | 每架无人机各自一个单机 TD3——无集中式 critic、无参数共享 |

论文的核心是一个**负结果 / 可靠性结果**：考察"降低 **TD 目标偏差**（估计 Q 值与其自举目标之差的带符号值）是否真的能提升多无人机策略的性能与可靠性"。

## 二、核心结论（全部为实测，原始数据见 results/）

1. **偏差不预测性能。** 在主种子上，**带符号**偏差最小的方法（MUAV-STD3，−0.29）反而**绝对**偏差最大（均值 11.41，p90 为 41.71），性能也并未占优——它的"近零"来自正负误差相互抵消。因此论文**并列报告**带符号与绝对偏差。
2. **多种子退化主导了排名。** 5 个种子中有 **3 个**对**每一种方法**都退化为**完全饱和的控制器**（饱和率 1.000，一回合内只有 2–3 个不同动作向量），这正是回报聚集在 −2,250 附近的原因。
3. **没有任何方法能可靠到达目标。** **全队到达率在每一个 run 中都是 0.000**；个体到达率在四种方法中有三种为 0.31–0.57，MUAV-MTD3 为 0.000。在更严的 0.3 相对阈值下，平均任务成功率为 0.155（MATD3），其余方法为 0.000。
4. **编队与机间避碰在本任务中不是活跃约束。** 机间惩罚项在 8 个被评 run 中有 **4 个全程未触发**，其余 run 最多只占 **1.8%** 的步数；编队误差维持在**规定构型自身标准差**（37.7 m）的 **0.72–4.49 倍**。
5. **不主张已验证的避障。** 障碍半径 25 m、惩罚触发中心距 20 m，意味着惩罚是在无人机**已进入名义障碍体积之后**才触发。实测每回合：中心距小于 25 m 的"真实侵入" **2.28–13.08** 次、安全余量违反（25 m ≤ d < 45 m）**8.4–31.1** 次、惩罚触发 **1.4–7.9** 次。
6. **算子开销（RTX 4090）。** 单次每 agent 梯度更新：MADDPG 5.24 ms、MATD3 5.92 ms、MUAV-MTD3 6.77 ms、MUAV-STD3 7.11 ms → MUAV-STD3 / MATD3 = **1.20×**；一个完整训练步为 16.8–22.2 ms。理论上 critic 前向次数之比为 **5×**（5 个采样动作 × 2 个 critic，对比 1 个动作 × 2 个 critic）。
7. **鲁棒性实验测的是观测扰动，不是通信延迟。** 所谓"延迟"条件是对 actor **自身**观测的亚周期加性扰动（该 10 维观测**不含队友信息**），dropout 作用于障碍相对分量。

## 三、测试台设置（所有方法完全一致）

| 设置 | 取值 |
|---|---|
| 无人机数量 | 3 架 |
| 地图 | 500 m × 500 m |
| 障碍半径 / 惩罚触发中心距 | r = 25 m / d_pen = 20 m |
| 控制步长 | dt = 0.5 s，每回合 200 步 |
| 运动学 | 双积分模型，a_max = 5.0 m/s^2，v_max = 15.0 m/s |
| 观测 | 10 维局部观测（位置、速度、相对目标、最近 4 个障碍的相对量）——**不含队友信息** |
| 奖励 | 5 项：目标距离、障碍接近（小于 20 m）、机间接近（小于 20 m）、能量、每步惩罚——**不含编队成形项** |
| 训练 | 每个种子 1,500 回合；种子 **42、123、456、789、1024** |
| 超参数 | gamma = 0.99，tau = 0.005，batch 128，hidden 128，回放池 1e5，预热 5,000 步；beta_soft = 5.0，K = 5，beta_ME = 0.5 |

## 四、仓库结构

```
uav_marl/                 训练与评估代码
  agents/                 maddpg.py、matd3.py、muav_std3.py、muav_mtd3.py、td3.py
  env/uav_env.py          多无人机路径规划环境（奖励、运动学、几何）
  networks/actor.py       actor 网络
  utils/                  回放池、噪声
  train.py、multi_seed_train.py、train_td3*.py
  eval_*.py               偏差 / 分离指标 / 机间诊断 / 鲁棒性评估
  *_sweep*.py、ablation_study.py、retrain_*.py    敏感性研究与消融驱动脚本
  gen_figures_5seed.py    绘图脚本
scripts/                  当时实际运行的 shell 脚本（00–12）
verify/verify_outputs.py  对产出的结果文件做一致性校验
merge/merge_sweeps.py     扫描结果合并
results/                  论文各表、各图背后的实测输出
docs/                     复现说明（中文）+ 结果文件字典
```

## 五、快速开始

```bash
git clone https://github.com/LiChen631/A-Multi-Seed-Reliability-Benchmark.git
cd A-Multi-Seed-Reliability-Benchmark
pip install -r requirements.txt

# 冒烟测试（短程运行，非论文设置）
python uav_marl/train.py --algo muav_std3 --episodes 50 --seed 42

# 论文设置下的单个"方法 + 种子"完整训练（1,500 回合）
python uav_marl/multi_seed_train.py --algo muav_std3 --seed 42 --episodes 1500

# 按论文协议做评估
python uav_marl/eval_qbias_fixed.py        # 带符号 + 绝对 TD 目标偏差
python uav_marl/eval_metrics_split.py      # 分离任务指标（到达、碰撞、安全）
python uav_marl/eval_interagent_diag.py    # 机间耦合 / 编队诊断
python uav_marl/eval_robustness_multi.py   # 观测扰动鲁棒性
```

逐步复现协议见 **docs/REPRODUCE_zh.md**；
"哪个结果文件对应论文哪张表/图"见 **docs/DATA_DICTIONARY.md**。

## 六、仓库中包含的结果文件

| 文件 | 内容 | 对应论文位置 |
|---|---|---|
| `results/qbias_fixed_results.json` | 各算法/种子的带符号与绝对 TD 目标偏差（均值、p50、p90） | 表 3、图 2 |
| `results/robustness_multi_seed.json` | 观测扰动下的回报（加性噪声 / dropout），5 个种子聚合 | 表 6、图 5 |
| `results/robustness_multi_seed.before_rerun.json` | 同一实验重跑前的版本，保留以溯源 | — |
| `results/metrics_split.json` | 分离任务指标：个体到达、全队到达、真实碰撞、安全违反、惩罚触发 | 表 9、第 6.7 节 |
| `results/interagent_diag.json` | 机间耦合频率、编队误差、编队质心距离 | 表 10、第 6.7 节 |
| 云服务器实测数据_20261001.json | 评估服务器上的实测汇总：5 种子退化、分离指标、耗时、偏差 | 表 3/5/6/8、第 4.4 节 |
| `results/log_robustness_5seed.txt` 等日志 | 原始运行日志 | 审计留痕 |

## 七、复现说明

* **未包含模型权重**（约 400 MB 的 .pt 文件）。训练可用 uav_marl/ 与 scripts/ 中的种子**完全复现**；若只做评估复现，则需要 checkpoint，可向作者索取。
* results/ 中的文件是论文全部数值的**唯一依据**；图由这些文件重新生成（gen_figures_5seed.py）。
* 耗时测量所用硬件：NVIDIA RTX 4090、16 vCPU Xeon Gold 6430、120 GB 内存、PyTorch 2.5.1+cu124、CUDA 12.4。
* 全部结论仅限本文使用的 **3 机、二维、质点**设定；论文明确**不外推**到更大机群、三维动力学或实机平台。

## 八、范围与诚实的局限

本仓库发布的是**基准与负结果**，**不主张新算法**。两种算子（softmax、互估计）取自已发表工作 [21]；本文的贡献是实例化、可靠性基准、失效模式分析与公开发布的测量数据。论文中的理论分析以**待检验假设**的形式呈现，而非已验证结论。

## 九、引用

```bibtex
@article{li2025tdtarget,
  title  = {TD-Target Discrepancy Does Not Predict Multi-UAV Policy Performance: A Multi-Seed Reliability Benchmark},
  author = {Li, Chen and Huang, Zikai},
  year   = {2025}
}
```

## 十、联系

Chen Li —— 代码或数据相关问题请在本仓库开 Issue。
