# 重跑实验包 · 运行说明（GPU 云服务器版）

> 目标：把审稿意见里"必须重跑"的部分跑完，并把结果交回给我回填论文。
> **重要前提**：这套 RL 代码的瓶颈是 **Python/NumPy 环境循环 + 大量小算子（网络只有 128×128）**，
> 不是大矩阵运算。所以：**选"GPU + 多核 CPU"的机器，并且要靠多进程并行**；单进程加 GPU 大约只有 3–5× 提升。

---

## 0. 包里有什么

```
重跑实验包/
├── README_运行说明.md          ← 本文件
├── uav_marl/                   ← 代码副本（已把写死路径改为读环境变量 UAV_RESULTS_DIR）
│   └── eval_metrics_split.py   ← 新增：分离统计真实碰撞/安全违反/惩罚触发 + 个体到达/无碰撞成功
├── scripts/
│   ├── 00_setup_check.sh       ← 上机自检（GPU/python/依赖/路径/冒烟测试）
│   ├── 01_fix_seed456_robustness.sh  ← 【最要紧】修 Table 6 的数据串档
│   ├── 02_eval_metrics_split.sh      ← 意见 #2/#4 的指标分离
│   ├── 03_eval_qbias_absolute.sh     ← 意见 #1 的偏差逐样本/绝对误差
│   ├── 04_timing.sh                  ← 补测 §4.4 的步时（现无出处）
│   └── 10_multiseed_sweeps.sh        ← 意见 #11 的同组多 seed 扫描（隔离目录，不覆盖）
├── merge/merge_sweeps.py       ← 把逐 seed 扫描结果合并成 mean±std 汇总
└── verify/verify_outputs.py    ← 结果校验（含"seed456 不得再与 MUAV-MTD3 相同"的硬断言）
```

**准备上传的东西**（在本地打包：`uav_marl/` + `scripts/` + `merge/` + `verify/`，约 0.5 MB）：

| 批次 | 需要上传的 checkpoint（放在 `results/checkpoints/`） | 体积 |
|---|---|---|
| **A（评估层，推荐先跑）** | `maddpg`、`maddpg_seed123`、`maddpg_seed456`、`matd3`、`matd3_seed123`、`matd3_seed456`、`muav_mtd3`、`muav_mtd3_seed123`、`muav_mtd3_seed456`、`muav_std3`、`muav_std3_seed123`、`muav_std3_seed456`（12 个目录） | ≈ 0.8 GB |
| A+（若要同时重算 5-seed 表） | 再加各算法的 `_seed789`、`_seed1024` 与 `td3`、`td3_seed123/456/789/1024`（13 个目录） | ≈ +0.9 GB |
| **B（重训层）** | **不需要**旧 checkpoint（从零训练） | — |

---

## 1. 租服务器怎么选（以 AutoDL 为例，恒源云/潞晨/vast.ai 同理）

| 项目 | 建议 | 原因 |
|---|---|---|
| GPU | **RTX 4090 / 3090 / A10 单卡** 即可 | 网络很小，算力不是瓶颈；不必上 A100（贵且浪费） |
| CPU | **≥ 8 vCPU**（16 更好） | 真正决定速度：每进程 1 核，8 核=8 进程并行 |
| 内存 | ≥ 32 GB（最低 16 GB） | 每个训练进程约 1–1.5 GB（replay buffer 5e5 条） |
| 磁盘 | ≥ 80 GB 数据盘 | checkpoint 每个 62 MB；批 B 的扫描输出约 5–10 GB |
| 系统盘/镜像 | **PyTorch 2.x + CUDA 12.1/12.4 官方镜像** | 免得自己装 torch（装 torch 最费时间）。⚠️ **镜像 CUDA 版本必须 ≤ 该机驱动支持上限**：驱动 560 → 只选 ≤12.6（cu121/cu124）；驱动 595 → 可选更新版本。选错会报 `driver version is insufficient to run this CUDA version`，上机第一件事就是用 `00_setup_check.sh` 确认 `cuda available: True` |
| 计费 | **按量计费**，先用最小配置试通流程再扩容 | 批 A 只需 1–3 小时 |
| 区域 | 与你网络近的（如 AutoDL 内蒙/内蒙B、恒源云华北） | 上传 0.8 GB 更快 |

**费用粗算**（按 ¥2/h）：批 A ≈ **¥5–10**；批 B（重训 55+ runs，8 进程并行约 20–30 h）≈ **¥40–80**。建议先充 ¥50 试批 A。

---

## 2. 上机步骤（照抄即可）

```bash
# ① 登录（AutoDL 控制台复制 SSH 命令）
ssh -p <端口> root@<主机>

# ② 传代码与 checkpoint（本机执行；用平台自带的 JupyterLab 上传也行）
#    在本地先打成两个压缩包：
#    tar -czf code.tar.gz uav_marl scripts merge verify
#    tar -czf ckpt.tar.gz results/checkpoints/{maddpg*,matd3*,muav_mtd3*,muav_std3*}
#    ⚠️ AutoDL 的数据盘挂载点是 /root/autodl-tmp（系统盘只有 30 GB，别放系统盘）
scp -P <端口> code.tar.gz ckpt.tar.gz root@<主机>:/root/autodl-tmp/

# ③ 解包（工作目录放数据盘）
cd /root/autodl-tmp && mkdir -p work && cd work
tar -xzf ../code.tar.gz && tar -xzf ../ckpt.tar.gz
mkdir -p results && mv checkpoints results/ 2>/dev/null || true
export UAV_RESULTS_DIR=/root/autodl-tmp/work/results

# ④ 装依赖（镜像一般已带 torch；只补 matplotlib）
pip install -q matplotlib
bash scripts/00_setup_check.sh        # 自检：GPU/依赖/路径/冒烟测试

# ⑤ 用 tmux 保持会话（断线不停任务）
tmux new -s run
```

> 把 `export UAV_RESULTS_DIR=/root/autodl-tmp/work/results` 写进 `~/.bashrc`，避免每次重设。
> **并行度按 16 核设 8–12 个进程**（留 2–4 核给系统与 GPU 喂数据）；120 GB 内存不是瓶颈（每进程约 1.5 GB）。

---

## 3. 分批运行计划

### 批次 A：修数据错误 + 补评估层指标（1–3 小时，建议先做）

```bash
# A1【最要紧】修复 Table 6 的 seed456 串档（约 10–40 分钟）
bash scripts/01_fix_seed456_robustness.sh

# A2 分离统计：真实碰撞 / 安全距离违反 / 惩罚触发 + 个体到达 / 无碰撞成功（约 1–2 小时）
bash scripts/02_eval_metrics_split.sh

# A3 偏差的逐样本/绝对误差（约 1–2 小时）
bash scripts/03_eval_qbias_absolute.sh

# A4 步时补测（几分钟）
bash scripts/04_timing.sh

# A5 全部校验（必须 PASS 才继续）
python verify/verify_outputs.py --results-dir "$UAV_RESULTS_DIR" --check all
```

**A1 的合格标准**（脚本会自动断言）：新的 `robustness_multi_seed.json` 中
`muav_std3.seed456` 与 `muav_mtd3.seed456` 的 12 个条件**不再逐个相同**（原来 9/12 逐位相同）。
若仍相同 → 把日志发我，说明评估脚本里还有别的串档来源。

### 批次 B：重训层（需先让我打 2 个补丁；见第 5 节）

```bash
# B1 同组多 seed 扫描/消融（11 配置 × 3 seeds = 33 runs；
#    串行约 165 h，8 进程并行约 20–25 h —— 用并行脚本，见下）
bash scripts/10_multiseed_sweeps.sh

# 并行跑法（把不同 seed 分配到不同进程；每个 seed 一个独立 results 目录）
for S in 42 123 456; do
  UAV_FORCE_SEED=$S UAV_RESULTS_DIR=$PWD/results_seed$S \
    bash -c 'cd uav_marl && python beta_sweep_v3.py --beta 0.1 0.5 1.0 5.0 10.0 --episodes 1500' \
    > log_beta_seed$S.txt 2>&1 &
done
wait
```

> 真实并行度取决于核数与内存：**8 进程 ≈ 8 倍吞吐**。开跑前先 `nvidia-smi`、`free -g`、`nproc` 看一眼。

---

## 4. 跑完要交回给我的文件

请把这些打包发我（或放到工作区/网盘）：

| 文件 | 用途 |
|---|---|
| `results/robustness_multi_seed.json` | 修 Table 6（MUAV-STD3 列） |
| `results/metrics_split.json` | 新增"真实碰撞/安全违反/个体到达/无碰撞成功"表（意见 #2/#4） |
| `results/qbias_fixed_results.json`（+ 若有逐样本文件） | 偏差的绝对误差汇总（意见 #1） |
| `sweep_multiseed/sweep_multiseed_summary.json` | 意见 #11 的同组多 seed 参数结论 |
| `verify/*.txt`（各次校验输出） | 我核对是否达标 |
| 各脚本的 `log_*.txt` | 审计与排查 |

---

## 5. 还需要我打补丁才能跑的三项（它们必须改代码，不适合匆忙上线）

| 项目 | 需要的代码改动 | 预计新增成本 |
|---|---|---|
| **意见 #3 有/无队友观测对照** | `env/uav_env.py` 增加可选队友相对位置（obs 10 → 16 维）并相应改 `agents/*` 的输入维度；新增训练入口跑 2 组 × 5 seeds | 10 runs ≈ 50 CPU-h（8 进程并行约 7 h） |
| **意见 #7 真实滞后观测** | 新写 `eval_lagged_obs.py`：维护带时间戳的观测环形缓冲，按 delay 取历史观测（非加性噪声）；只评估 | 评估约 2–3 h |
| **意见 #8 梯度方差测量** | 在 `train.py` 的更新处记录 actor/critic 梯度范数与 loss，落盘 JSON；重跑 4 算法 × 3 seeds | 12 runs ≈ 60 CPU-h（并行约 8 h） |

这三项的补丁我可以直接写好放进包里（下一轮），**先跑批 A** 不影响它们——批 A 与它们互不依赖。

---

## 6. 常见坑

1. **断点续跑**：`multi_seed_train.py` 与三个扫描脚本都支持 resume（会打印 `[SKIP]`/`[RESUME]`），
   tmux 里中断后可原地重跑，不会白干。
2. **别用 `nohup ... &` 不管输出**：日志要落盘（脚本已 `tee`），否则出问题无法排查。
3. **matplotlib 后端**：无显示环境必须 `matplotlib.use('Agg')`（包内脚本已设置）。
4. **CUDA/torch 版本**：`config.py` 自动 `cuda if available else cpu`；若 `torch.cuda.is_available()` 为 False，
   说明镜像与驱动不匹配，换 PyTorch 镜像即可，不要自己 pip 装 torch（很慢）。
5. **多进程与显存**：网络小，显存占用低（<2 GB/进程）；但**每个进程都会占 CPU 核**，别开超过核数。
6. **路径**：全部通过 `UAV_RESULTS_DIR` 控制；脚本会拒绝写到你没设置的目录（默认 `../results`）。
7. **随机性**：`eval_robustness_multi.py` 会在每个 `(algo, seed)` 开始时 `np.random.seed(seed)`，
   因此**不同算法的障碍布局是同一串**，可以直接对比（这也是判断"串档"的依据）。
8. **上传慢**：0.8 GB 走 scp 可能要几分钟；平台 JupyterLab 上传对 1 GB 以内更快。
9. **不要覆盖原始结果**：脚本会把旧文件备份成 `*.before_rerun.json`；新结果写回同名文件。
10. **校验先于下载**：`verify_outputs.py` 不过就别下载，先看日志（我也可以帮你看）。

---

## 7. 最小行动清单（如果你只想花 1 小时、¥5）

```bash
bash scripts/00_setup_check.sh
bash scripts/01_fix_seed456_robustness.sh     # 修掉唯一的数据错误
python verify/verify_outputs.py --results-dir "$UAV_RESULTS_DIR" --check seed456_duplication
# 把 robustness_multi_seed.json 和校验输出发我
```
