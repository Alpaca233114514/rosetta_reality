# Rosetta Reality M2 受控实验工作日志与新任务交接

日期：2026-08-09（Asia/Shanghai）
状态：**M2 未完成；research only；不得用于真机。**

## 0. 新任务从这里开始

本文件是当前任务的唯一恢复入口。用户要求当前任务不再开启新训练，因此工作在以下边界停止：

- v004 Base absolute：正式训练、resume、export/reload、validation、Gate 3/4 已完成；Gate 4 失败。
- v005 Base residual：正式训练、resume、export/reload、validation、Gate 3/4 已完成；候选被拒绝。
- v006 Instruct control：Gate 1/2、完整 frozen feature cache、pre-training benchmark 已完成；**optimizer 从未启动**。
- v006 下不存在 `smoke/`、`overfit/`、`training/` 或 `evaluation/`。
- v004、v005、v006 的 hidden test 均未打开；不得为了选模型读取 test。
- 没有启动 v007、DAgger 或 RL；没有下载新模型或新数据。
- 没有删除文件，没有 commit，没有 push，没有修改 `main`/`master`。
- 最终封口时 Docker daemon 已不可连接；WSL 中无 cache/benchmark/train/sim/pytest 残留进程。
  新任务继续任何容器工作前必须重新启动 Docker，并重新运行只读 environment/container check。

当前 Git 状态：

- branch：`codex/m2-qwen08b-frozen-001`
- HEAD：`c0aa0bd0490c6655c098ccf73b9bc531fa9d9c96`
- workspace：dirty；缓存 v006 时记录 123 个受身份约束文件
- v006 cache 所绑定 workspace tree SHA-256：
  `0c5a87d97549e96da44c12d9de9ab3a273011795e03216a923351120366257e9`

不要把 dirty workspace 的 branch 名称当作完整代码身份；以各 manifest 中的 HEAD、dirty 标志、
workspace tree SHA 和文件数共同约束。

## 1. 技术结论

### 1.1 已证实

1. Base 身份已经纠正并固定为 `Qwen/Qwen3.5-0.8B-Base`，revision
   `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`。正式 M2 仍只能由 Base 实验满足。
2. v004 在 Intel XPU 上复现了 CPU spatial reference：validation MAE
   `0.0266666878`，与此前 CPU 值 `0.0266664755` 的差约 `2.12e-7`。因此设备切换不是失败主因。
3. 数据确实需要视觉信息：validation 的 5 个 reset state 两两完全相同，目标首动作两两 L2
   均值却为 `0.112095438`。仅由 robot state 不可能判定正确首动作。
4. v004 使用了视觉，但响应不足：reset 预测动作两两 L2 均值为 `0.047029365`，只有目标差异的
   `41.95%`。
5. v005 residual 把 validation MAE 从 `0.0266667` 降到 `0.0195459`，但同时降低视觉依赖并显著
   恶化闭环安全：Gate 4 raw-limit violation、joint-limit violation 和碰撞都大幅上升，task
   success 仍为 0%。这是明确的“offline 更好、closed-loop 更坏”。
6. v004 与 v005 均为 0/5 task success、maximum reward 0。因此 M2 task capability 未通过，不能
   宣称 M2 完成。
7. Instruct 权重在工作区外已有本地副本；本任务离线采用并校验，没有下载。v006 只用于
   backbone/prompt 对照，配置明确写有 `m2_completion_eligible: false`。

### 1.2 高概率、但尚未直接证实

当前主假设是 behavior cloning 的 policy-induced distribution shift / compounding error：

```text
expert-distribution 上的小误差
        -> policy 进入训练集中较少见的 state
        -> 预测进一步偏离
        -> 无恢复能力
        -> 越界 / 碰撞 / 任务失败
```

现有 Gate 4 只保存 episode 聚合指标，没有逐步 state/action trace，所以它证明“闭环失败”，尚未
直接证明“首次偏离发生在哪一步、误差如何增长”。不得把这一主假设写成已完成的因果验证。

### 1.3 现在不应直接上 RL 或伪 DAgger

- Gate 4 reward 很稀疏，Base policy 还没有非零 task success；直接 RL 会把尚未分清的视觉、
  action、distribution-shift 问题一起塞进更难调的优化器。
- 已有 expert replay 不是任意 policy state 的 recovery oracle。policy 偏离以后，同一时间索引的
  dataset action 只是一条参考动作，不能冒充该偏离 state 的正确标签。
- 只有获得可信的 state-conditioned oracle/controller 后，才能把 policy 访问到的偏离状态回灌为
  DAgger 数据。

## 2. 受控实验轴

| 实验 | Backbone | Prompt | Adaptation | Feature | Action parameterization | Training device | 当前状态 |
|---|---|---|---|---|---|---|---|
| v004 | Qwen3.5-0.8B-Base | base multimodal | frozen | final hidden, image spatial 2x2 | absolute | Intel XPU | 完整训练；Gate 4 failed |
| v005 | 与 v004 相同 | 与 v004 相同 | frozen | 与 v004 相同 | residual from current state | Intel XPU | 完整训练；rejected |
| v006 | Qwen3.5-0.8B Instruct | native chat template | frozen | final hidden, image spatial 2x2 | absolute | cache: Intel XPU | cache + benchmark only；M2-ineligible |

v004→v005 的唯一研究轴是 `action_expert.prediction_parameterization`。
v004→v006 的唯一研究轴是 backbone checkpoint variant 与其原生 prompt protocol。v006 的数据、
split、action contract、下游维度、optimizer 配置、seed、batch、epoch、early stopping 和 XPU 资源
均保持一致。

配置文件：

- `configs/experiments/m2_qwen08b_frozen_004_spatial_xpu_control.yaml`
- `configs/experiments/m2_qwen08b_frozen_005_spatial_residual_xpu.yaml`
- `configs/experiments/m2_qwen08b_frozen_006_instruct_spatial_xpu_control.yaml`

## 3. 模型、数据和 Action Contract

### 3.1 正式 Base

- identifier：`Qwen/Qwen3.5-0.8B-Base`
- revision：`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- local model manifest：
  `models/Qwen--Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/model_manifest.json`
- manifest SHA-256：`08da02c07081c28eb9c6605de182924fa4be97d2b3f47d5c37283cab267e2c63`
- weight SHA-256：`c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c`

### 3.2 辅助 Instruct control

- identifier：`Qwen/Qwen3.5-0.8B`
- revision：`2fc06364715b967f1860aea9cf38778875588b17`
- adopted local model manifest SHA-256：
  `d598388455a73f2d37d739f6e5ecd7048420435375824f737bd633e43cb3f6da`
- weight SHA-256：`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
- tokenizer SHA-256：`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
- 来源副本存在 provider metadata 差异；运行时所需文件与配置声明的 upstream digest 均已离线验证。
- 不要把 Instruct 结果用于满足 Base-only M2 completion gate。

### 3.3 Dataset

- repo id：`lerobot/aloha_sim_insertion_human`
- immutable revision：`cc571a3c661df81b566dbfde3d5c1e85fcdf7884`
- 本地使用全部 50 个 episode，没有可通过“再打开现有 episode”扩展的余量。
- 固定 split seed：`20260809`
- train：40 episodes / 3960 cached samples
- validation：episodes `[22, 13, 7, 33, 45]` / 495 samples
- hidden test：episodes `[31, 6, 1, 24, 5]` / 495 samples
- frame stride：5
- instruction：`Insert the peg into the socket.`
- dataset config：`configs/data/aloha_sim_insertion_m2.yaml`

v004–v006 的 test shard 随不可变 cache 一并存在，但 benchmark/training/validation diagnostics 不加载它。

### 3.4 Rosetta Action Contract

- config：`configs/sim/aloha_insertion.yaml`
- 14 维双臂 joint-space absolute position action
- 50 Hz
- chunk length 8
- formal execution：`receding_horizon_first_action`
- 输出先检查 finite，再按合同裁剪；raw 和 executed violation 分开记录
- 左右臂、各关节和 gripper ordering 以 config 为唯一依据，不能仅凭 shape 判断兼容

Gate 1 和 Gate 2 已证明 adapter、ordering、方向、范围以及选定 expert replay 的基本语义可工作。
Gate 2 的 episode 2 / seed 10 在 294 个 expert action 后达到 success、maximum reward 4，initial image
MAE `0.0013727`，expert tracking mean MAE `0.0435047`。

完整 expert replay 证据的准确口径：episode 1–4、最多 500 个 expert action，并在轨迹未结束时允许
最多 100 个末端 settle；结果为 3/4 success、无 calibrated unexpected collision。episode 3 达到
reward 3 后超时。历史 `maximum_steps=1` 的 5-episode report 只是 seed/image 对齐探针，不能当作
0/5 完整 expert replay。

## 4. 开发环境、Docker 与资源隔离

### 4.1 宿主和容器运行面

- Windows 11，version `10.0.26200`，build `26200`
- WSL `2.7.11.0`
- WSL kernel `6.18.33.2`
- Docker client/server `29.6.1`，API `1.55`，Linux amd64
- Intel(R) Graphics，Windows driver `32.0.101.8801`，driver date 2026-05-12
- XPU 容器识别设备：`Intel(R) Graphics [0x7dd1]`
- 本任务没有安装或升级 Windows/WSL GPU driver、system CUDA、ROCm、system PyTorch 或 simulator。

### 4.2 精确 Docker image identity

| Role | Dockerfile | Image digest |
|---|---|---|
| ML CPU | `docker/Dockerfile.m2` | `sha256:c107d41f49b92b1a8a8f0c7848802b860cbbe467eae698253e1a76d648f0e3d6` |
| ML XPU | `docker/Dockerfile.m2-xpu` | `sha256:449ef8e059edaae7b88499ab6dd90ee3d376be841258c3a5c6eeac26d31fef74` |
| Sim CPU | `docker/Dockerfile.sim` | `sha256:fcacf7315c6880ba97d11a95af67e4c54c0cc7d9e76af0a93b61bd0520bd9f37` |
| Sim XPU | `docker/Dockerfile.sim-xpu` | `sha256:f5227bffe19f3c1e8e6c0e2e0fa9699998a7eac718acde26a9ff10ff157163f9` |

构建与入口文件 SHA-256：

- `docker/Dockerfile.m2`：`f27dec849e15d3fbf85b0afe39dfd654d80b9bf1861a2e0cec0623dad8beada5`
- `docker/Dockerfile.m2-xpu`：`09524dbc31796f62f5b2b5b944cc8536c83f716a8c4ee488fa1d65b9cb0945e7`
- `docker/Dockerfile.sim`：`c0aeacf35d43098b5c36e991fe8d87400d108dbd65905f0b9d9c4d171e494e9c`
- `docker/Dockerfile.sim-xpu`：`5febe64e653ff5c01e01d38706120d3ee1df0e602605f5a4659ea42a4033ed03`
- `scripts/run_m2_container.sh`：`9e22a0318990d208dc5e7333b2a8f5317fd7c9ced0f67c12d537b3614b239e70`

### 4.3 关键包版本

| Role | Python | PyTorch | Transformers | PEFT | LeRobot | NumPy | 其他 |
|---|---:|---:|---:|---:|---:|---:|---|
| ML CPU | 3.13.5 | 2.11.0+cpu | 5.14.1 | 0.20.0 | 0.6.1 | 2.2.6 | pyarrow 25.0.0; safetensors 0.8.0; accelerate 1.14.0; Pillow 12.2.0 |
| ML XPU | 3.12.3 | 2.11.0+xpu | 5.14.1 | 0.20.0 | 0.6.1 | 2.2.6 | pyarrow 25.0.0; safetensors 0.8.0; accelerate 1.14.0; Pillow 12.2.0 |
| Sim CPU | 3.11.2 | 2.11.0+cpu | 5.14.1 | 0.20.0 | not installed | 2.4.4 | gym-aloha 0.1.4; gymnasium 1.3.0; MuJoCo 3.8.1 |
| Sim XPU | 3.12.3 | 2.11.0+xpu | 5.14.1 | 0.20.0 | 0.6.1 | 2.2.6 | gym-aloha 0.1.4; gymnasium 1.3.0; MuJoCo 3.8.1 |

### 4.4 实测隔离条件

`scripts/check_container.py` 在 exact ML CPU image 中返回 `status=passed`：

- memory hard limit：`5368709120` bytes（5 GiB）
- memory swap：0
- CPU quota：2 cores
- PIDs：512
- root filesystem：read-only
- `/workspace`：read-only
- 声明的 data / feature cache / runs 输出挂载：writable
- network interface：只有 `lo`
- `no-new-privileges=true`

通用入口是 `scripts/run_m2_container.sh`。关键环境变量：

- `ROSETTA_DOCKER_COMMAND=docker`：WSL 内使用 Linux Docker CLI
- `ROSETTA_MODEL_ROOT`：单个模型 snapshot，只读挂载到容器
- `ROSETTA_DATA_ROOT`、`ROSETTA_FEATURE_ROOT`、`ROSETTA_CHECKPOINT_ROOT`、
  `ROSETTA_ARTIFACT_ROOT`、`ROSETTA_RUN_ROOT`：限定生成物目录
- `ROSETTA_DOCKER_MEMORY=5g`
- `ROSETTA_DOCKER_CPUS=2`
- `ROSETTA_DOCKER_PIDS=512`
- `ROSETTA_XPU_DEVICE_PATH=/dev/dxg`
- `ROSETTA_WSL_LIB_ROOT=/usr/lib/wsl`

模型与数据处理、ML 测试、feature extraction、训练和仿真必须继续从 WSL Bash 进入这些容器；
不要用 Windows Python 执行。

## 5. 工作日志与门禁顺序

### 5.1 共同前置

- 完成 Base 身份修正和 revision pinning。
- 完成 CPU environment check、unit tests、dummy forward 和 CPU smoke。
- 完成 XPU environment check 和短 XPU smoke；Intel XPU 可用。
- 完成 Gate 1 scripted action 和 Gate 2 dataset replay。
- 每个 experiment 在任何 optimizer step 前创建与 feature identity 精确匹配的 immutable benchmark。
- hidden test 在 v004–v006 全程保持未打开。

### 5.2 v004 Base absolute XPU control

- cache identity：`bd3ee29d96a3da7ad465760460efa3073e074d2106aee3bb4752124ea059fecb`
- optimizer smoke：passed
- 32-sample overfit：passed；final/initial loss ratio `0.000637`
- run id：`m2-qwen08b-frozen-004-xpu-001`
- intentional stop：epoch 5；随后从 checkpoint 恢复
- completed epoch：40；not early-stopped
- best epoch/checkpoint：40 / `epoch-040.pt`
- checkpoint SHA-256：`7dea8166472222a76ae487533a7d67d49dd1d23dfbc2768fb003dbc239022f02`
- trainable downstream parameters：1,279,600；backbone frozen
- export/reload：exact match
- artifact：
  `artifacts/m2-qwen08b-frozen-004-spatial-xpu-control/m2-qwen08b-frozen-004-spatial-xpu-control-base-dc7cdfe2`

### 5.3 v005 Base residual XPU

- cache identity：`47f6cc227b4371c8a719ddbfaf38d959297c62d1c08990e670177b322a970b8d`
- optimizer smoke：passed
- 32-sample overfit：passed；final/initial loss ratio `0.001634`
- run id：`m2-qwen08b-frozen-005-residual-xpu-001`
- intentional stop：epoch 5；随后从 checkpoint 恢复
- completed epoch：22；early-stopped
- best epoch/checkpoint：14 / `epoch-014.pt`
- checkpoint SHA-256：`0272fb183a274bf0ca3980f657beb567afd2b7f2e45de2888fe3c57b0094056b`
- trainable downstream parameters：1,279,600；backbone frozen
- export/reload：exact match
- artifact：
  `artifacts/m2-qwen08b-frozen-005-spatial-residual-xpu/m2-qwen08b-frozen-005-spatial-residual-xpu-base-dc7cdfe2`

### 5.4 v006 Instruct control：当前硬停止点

- experiment role：`auxiliary_backbone_control`
- `m2_completion_eligible: false`
- full feature cache identity：
  `7e0bb793c1769cc6ce795d38d375eeeed96c1d27f627e35782633c3fec44f145`
- cache status：complete
- extraction device：Intel XPU
- cache dtype：float16
- pooling：image spatial 2x2；feature dimension 4096
- 50 episode 全部完成；train 3960 / validation 495 / test 495
- elapsed：3238.0 seconds
- extraction 期间容器约 3.94–3.95/5 GiB；GPU compute 实测约 83%；network I/O 0
- cache manifest 完成时间：2026-08-09 20:11:12 +08:00
- pre-training benchmark 完成时间：2026-08-09 20:12:35 +08:00
- benchmark `hidden_test_loaded=false`
- optimizer smoke：未运行
- overfit：未运行
- formal training：未运行
- artifact/evaluation/Gate 3/Gate 4：未运行

v006 benchmark（validation 495 samples）：

| Baseline | Action MAE | Action RMSE | Raw limit violation rate |
|---|---:|---:|---:|
| current-state persistence | 0.057019465 | 0.139468074 | 0.003751804 |
| train action mean | 0.177332476 | 0.233800322 | 0 |
| deterministic untrained policy | 0.182554722 | 0.240551904 | 0 |

这些值只证明 benchmark 已固化，不能代表 Instruct 对照的训练后性能。

## 6. v004 / v005 同协议结果

| 指标 | v004 Base absolute | v005 Base residual |
|---|---:|---:|
| Validation chunk MAE | 0.026666688 | 0.019545944 |
| Validation first-action MAE | 0.022433855 | 0.014939558 |
| Mean-feature ablation MAE delta | 0.030041834 | 0.006430900 |
| Same-frame cross-episode shuffle MAE delta | 0.019589903 | 0.004021442 |
| Reset predicted pairwise L2 | 0.047029365 | 0.031708047 |
| Reset target pairwise L2 | 0.112095438 | 0.112095438 |
| Reset response ratio | 0.419548 | 0.282867 |
| Validation raw-limit rate | 0.010191198 | 0.020815296 |
| Gate 3 | passed | passed |
| Gate 4 task success | 0/5 | 0/5 |
| Gate 4 maximum reward | 0 | 0 |
| Gate 4 raw-limit rate | 0.020257143 | 0.094085714 |
| Gate 4 joint-limit violations | 68 | 771 |
| Gate 4 calibrated unexpected collisions | 0 | 518 |
| Gate 4 status | failed | failed |

v005 的 offline MAE 改善约 26.7%，但视觉打乱敏感度、reset 响应比和闭环安全全部变差。
因此 residual hypothesis 被证伪，v005 不得选为下一阶段 checkpoint。

v004 的 68 次 joint-limit violation 也意味着它没有通过 safety execution status；即使 corrected
unexpected collision 为 0，也不能标记安全通过。两者都没有 task capability。

## 7. 碰撞指标边界

早期实现把同一 arm 内左右 gripper finger 的常驻接触误报为异常：静止 reset/no-op 每帧固定
8 个 contact points。修正后只排除“两个 geom 都是 gripper finger 且属于同一 arm namespace”的
pair；不同 arm 的 gripper-gripper、其他 robot-robot、非夹爪 robot-table/object 仍计异常。

因此：

- 历史未校准 collision count 不能作为危险严重度证据。
- v004/v005 最新 Gate 4 使用 corrected classifier 和 canonical pair 记录。
- v005 的 518 次包含真实 table/wrist/gripper 等 unexpected pairs，不是旧的同臂夹爪基线。
- task failure 独立由 0/5 success 和 maximum reward 0 证明。

## 8. “第三炉炸了”上下文后的独立判断

补回的对话上下文把问题描述为“做题模型”、behavior cloning closed-loop distribution shift，并建议
优先考虑 rollout→oracle recovery labels→DAgger，再考虑 RL。工作区证据支持这一方向，但还要补一层
直接验证。

下一项最有辨识力的只读诊断应为：

1. 只使用 validation episode，不打开 hidden test。
2. 为 validation 初始图像搜索对应 simulator seed，并验证 reset image/state alignment。
3. 同一 seed 启动两个独立环境：expert 环境执行合同裁剪的 dataset action；policy 环境闭环执行
   模型首动作。
4. 逐步记录 expert/policy state、raw/clipped action、reward、joint-limit、collision。
5. 报告 state MAE/L2 在固定阈值 `0.01/0.025/0.05/0.1` 的首次 crossing，不事后挑阈值。
6. 报告首次 raw-limit、joint-limit、unexpected collision 和 reward divergence 的 step。
7. policy 偏离以后，将 dataset action 标为 `time-indexed expert reference`，明确它不是 recovery oracle。
8. 先做 3-step smoke 证明双环境 reset 一致，再跑完整 trajectory。

现有 `scripts/sim_gate.py execution-diagnostic` 只比较 chunk execution strategy，仍只输出聚合数，不能
回答首次偏离问题。当前任务没有实现上述 trace，以免在 v006 cache 生成中改变 workspace identity，且
用户随后要求停止新实验。

## 9. 新任务执行清单

### 9.1 首先复核，不训练

1. 阅读本文件和 `AGENTS.md`。
2. 核对 branch、HEAD、dirty workspace；保护现有用户改动，不清理、不 reset。
3. 核对 v006 cache manifest SHA 和 benchmark SHA。
4. 确认 v006 仍不存在 `smoke/`、`overfit/`、`training/`、`evaluation/`。
5. 使用本文件的 exact Docker image digest；模型 snapshot 只读挂载；保持 `--network none`。
6. 在实际开始 v006 optimizer 前，按仓库规则重新确认 `check_env`、unit tests、dummy/CPU smoke 和短
   XPU smoke 的当前证据。不要安装 driver 或 system packages。

### 9.2 如果用户在新任务授权继续 v006 control

严格顺序：

```text
verify existing immutable benchmark
    -> one-step optimizer smoke
    -> 32-sample overfit
    -> formal train, intentional stop at epoch 5
    -> explicit resume
    -> export best checkpoint
    -> independent artifact reload
    -> validation-only eval and modality diagnostics
    -> selection decision
    -> Gate 3
    -> Gate 4 only if justified by the predeclared comparison
```

建议 run id：`m2-qwen08b-frozen-006-instruct-xpu-001`。
建议 artifact id：
`m2-qwen08b-frozen-006-instruct-spatial-xpu-control-instruct-2fc06364`。

pre-training benchmark 已存在，不要重写：

```text
runs/m2-qwen08b-frozen-006-instruct-spatial-xpu-control/benchmark/
pre-training-7e0bb793c1769cc6.json
```

训练器应由 matching benchmark identity 自动放行；不要绕过它。Instruct snapshot 继续使用已 adopt 的本地
副本并离线校验，不要下载。

训练后固定比较：

- validation MAE 和 train-validation gap
- mean-feature ablation delta
- within-frame cross-episode shuffle delta
- reset predicted/target response ratio
- raw-limit rate
- Gate 3 reload consistency
- 若跑 Gate 4：success、reward、raw/executed limit、joint limit、collision、smoothness、latency

即使 v006 成功，它也只是辅助对照；正式 M2 仍需 Base checkpoint 通过全部 gate。

### 9.3 数据扩展和下一代 Base

- 当前本地 dataset 已用全部 50 episodes；简单“再开已有 episode”不可行。
- 不得复制 episode、把 policy rollout 冒充 expert，或把偏离后的 time-indexed dataset action 当
  recovery label。
- 若要扩展，必须记录新轨迹来源、oracle/controller 身份、action contract、seed、revision、split 和
  leakage audit。
- 优先完成 trajectory trace，再决定是构建 recovery oracle、做 interactive imitation learning，还是
  设计新的 Base 单轴实验。
- RL 只应在基本 BC/DAgger 能获得非零、可重复 success，或有明确 dense/validated reward 与资源预算后
  进入。

## 10. 最终 QA

在 exact ML CPU image、断网、只读 workspace 中完成：

- `python -m ruff check .`：passed
- `python -m pytest -q`：`114 passed, 4 skipped`
- skip 1：CUDA unavailable
- skip 3：设计为需当前 real-data field mapping 的 tests
- warning 1：只读 workspace 无法写 `.pytest_cache`；不影响测试结果
- `scripts/check_container.py`：passed
- CPU `scripts/check_env.py`：passed；Python 3.13.5 / PyTorch 2.11.0+cpu
- XPU `scripts/check_env.py`：passed；Python 3.12.3 / PyTorch 2.11.0+xpu / 1 Intel XPU

## 11. SHA-256 证据索引

### 11.1 Config / cache / benchmark

| Artifact | SHA-256 |
|---|---|
| v004 config | `fb70740e0c15e94441e65b4c074d41c3bcb7eb49e8ffcbc2e0161f4e7134579c` |
| v005 config | `f6839b961a98d90180f95a2ca8b4fbf8d53a65a140e8d7a47bdbf592dea1b50c` |
| v006 config | `92f11d46822645f748f9b2b7e35bf535925bc75d8155145a0e59834c9517a003` |
| v004 cache manifest | `dd452b6b46ac6f572717a1cc09b7f886e2e8f62f8cc93690b88df9d3aaa69d51` |
| v005 cache manifest | `91f409bbbc130d6070bc27a0cf21f7c2e8c470436ec3b7bd7cd83314b144341b` |
| v006 cache manifest | `3ef0ef5d0347faa6705220799b16f9f2ea5091d1a73dfb7ed7198ddd3ac9ee65` |
| v004 benchmark | `e4be48e1c0f2b11ef5bde68a8bb560df6de7bbcb88b2cfc72e76c837129eefa9` |
| v005 benchmark | `6e35b375434649a6e5100b1aeb9e9698170ebb07f3c251afd5f3ad00a7776fbd` |
| v006 benchmark | `f3573ef98f2cf8a541f7f474a304fddff4f3392cf533ebb4b922027eb3c4281c` |

### 11.2 Training / artifact

| Artifact | SHA-256 |
|---|---|
| v004 training manifest | `53e06e5b49fbd862dd77fcdbf0af426fc4cd8b9bdcc846ca03a75d39d19b1eb5` |
| v005 training manifest | `c30f15dec692f947de15d3ab4431d11e09ca36f1434a7d971c4ca33d159fa372` |
| v004 exported artifact manifest | `4e56fa101227a5e60d9a6ad53c9c5cf370168b7beff416e2d9bf9e4d6543aead` |
| v005 exported artifact manifest | `0fce9cc5701ba2611c90bcb5605c68437a62f15353d962afb10e17a99992a227` |

### 11.3 Eval / diagnostics / gates

| Artifact | SHA-256 |
|---|---|
| v004 validation eval | `a431d388192fa6e26a5a3caee3ddc25b9d80ce27508f460ece1335c9dd000309` |
| v005 validation eval | `18b12421bdde981245949190ad856cecfb53cd5464085cd7e059c3ff62916b5c` |
| v004 cached-policy diagnostic | `785d4076bcc17cb1d0d87d9a183288f22b0b509766ea9b6aced18eea92c800ac` |
| v005 cached-policy diagnostic | `04905cc8fc995a123aa876c7dca6d6179e00af9162fa22111cd56826b9762483` |
| v004 modality audit | `0a1ee7b8ec532309e03cb2ae3cf93f533f9a452bc6f8b40ad3ce4c5435268b28` |
| v005 modality audit | `b10d288f9c89353bccc209502c13f819ed00cf866bee339ac1e5fe4e8d8f3fce` |
| full expert replay | `d87a73fbd5f24b2b00be81c988632a814a9dddc544b8d14f5d8daa26a0741ab0` |
| v004 Gate 3 | `a4e175705440cc41047db282b733d8ca8eb82601d6d655c0a703c1a469126537` |
| v004 Gate 4 | `c005ca40514019af3dc82a77df1278610f1122e4742045d8334dd6d1bc54eeb9` |
| v005 Gate 3 | `a4a2fc70e57a5c0936b27f532e3073bed8dac279b608178e08a23d8d668e10bb` |
| v005 Gate 4 | `193dd243747e840150164b7e9bdb9d642ca8fdf792d023caf474e93a64418170` |
| v006 Gate 1 | `ea33318bb505c01b99dd9a654b07676c49325dec85a17af64d242c2098364816` |
| v006 Gate 2 | `045f474438322afc3c38abb43f32d6100ae598663005913867626bffeb2d041b` |

## 12. 最终发布判断

- 不发布为成功 insertion policy。
- 不上传 Hugging Face 或其他外部平台。
- 不部署真机。
- 不进入 9B scale-up。
- 不标记 M2 complete。
- v005 保持 rejected。
- v006 保持 untrained auxiliary control，等待新任务和用户授权。
