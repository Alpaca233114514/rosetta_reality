# Rosetta Reality M2 — Qwen3.5-0.8B-Base Frozen 实验报告

- 实验 ID：`m2-qwen08b-frozen-001`
- 记录日期：2026-08-09
- 分支：`codex/m2-qwen08b-frozen-001`
- 基准代码 revision：`c0aa0bd0490c6655c098ccf73b9bc531fa9d9c96`
- 实验工作区快照：`ba030d900c5e6abc0937d9c4cf8dfd5ae7aa23126b47c5c69f95704c02be59a9`
- 模型：`Qwen/Qwen3.5-0.8B-Base`
- adaptation：Frozen Backbone + Rosetta Projector / State Encoder / Fusion / Action Head
- 研究用途：experimental / research only；没有真机验证

## 技术结论

本次实验完成了正确 Base 模型的 revision-pinned 获取、数据验收、Action Contract、MuJoCo
Gate 1/2、冻结特征缓存、训练前 benchmark、optimizer smoke、小样本 overfit、完整训练、显式
checkpoint resume、validation、隐藏 test、artifact export/reload 以及 Gate 3/4 多回合闭环评估。

工程链路结论为“完成并可复核”，任务能力结论为“未通过”：最佳 validation MAE 为
`0.0230362`，相对预先声明的 train-action-mean benchmark `0.1773325` 降低 `87.01%`；隐藏
test MAE 为 `0.0291032`。经碰撞分类修正和完整 300-step horizon 复验后，Gate 4 的 5 个
episode、1,500 个控制 step 中任务成功率仍为 `0%`、maximum reward 仍为 0；修正后的
unexpected collision 为 0，但 raw action limit violation rate 为 `0.01905%`。因此当前 checkpoint 证明了完整
observation-action-observation 链路和约束后动作执行，但没有证明插销任务能力，不应进入 M3
9B scale-up，也不应对外声称任务已解决。

历史 Gate 4 把 JSON `status` 标成 `passed`，但旧判定没有使用任务成功率和碰撞数。审计修正后，
新协议把 finite/raw/executed action、joint limit、task success 和 calibrated unexpected collision
分别列为 acceptance criteria，预先要求 task success rate 至少 `20%` 且 unexpected collision 为 0。
新 Gate 4 的 `status`、`safety_execution_status` 和 `task_capability_status` 均为 `failed`。

## 实验假设与停止条件

假设：冻结的 Qwen3.5-0.8B-Base pooled multimodal representation 与归一化 robot state 融合，
能够在 held-out ALOHA insertion episode 上优于未训练 action head 及 train-action-mean baseline，
同时输出可被 Rosetta Action Contract 安全执行的动作。

受控实验轴：

| 实验轴 | 本次取值 |
|---|---|
| Backbone scale | `0.8B` |
| Backbone identity | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2...` |
| Adaptation | `frozen` |
| Action Expert | continuous MLP，493,168 trainable parameters |
| Dataset | `lerobot/aloha_sim_insertion_human@cc571a3c...` |
| Simulation | Gym-ALOHA / MuJoCo insertion |

停止条件包括 non-finite loss/action、Action Contract 或身份不匹配、容器资源上限、以及 validation
连续 8 个 epoch 不改善。完整训练在 epoch 39 因 early stopping 停止，最佳 checkpoint 为 epoch 31。

## 隔离环境与资源边界

所有模型、数据、训练、评估和 MuJoCo 命令均在 WSL Bash 启动的 Linux Docker 容器内运行。
Windows 侧仅做文件编辑、Git/静态只读检查与哈希复核。

| 项目 | 值 |
|---|---|
| ML image | `rosetta-reality-m2:local@sha256:c107d41f49b9...` |
| Simulation image | `rosetta-reality-sim:local@sha256:fcacf7315c68...` |
| 容器 memory / memory+swap | `5 GiB / 5 GiB`，即无额外 swap |
| CPU quota | `2` |
| PID limit | `512` |
| Root filesystem | read-only |
| Linux capabilities | all dropped |
| Security option | `no-new-privileges` |
| Offline stages | `--network=none`，HF datasets/model offline |
| 可写 mount | data、feature cache、checkpoint、artifact、run output |
| Model mount during ML | read-only |
| Runtime | Python 3.13.5，PyTorch 2.11.0+cpu，CUDA unavailable |

网络只在获得授权的数据/模型准备命令中开启。容器使用 `--rm`，最终验收后没有遗留运行中容器。

关键 package 版本：ML image 为 Transformers 5.14.1、PEFT 0.20.0、LeRobot 0.6.1、PyArrow
25.0.0、NumPy 2.2.6；simulation image 为 Gym-ALOHA 0.1.4、Gymnasium 1.3.0、MuJoCo 3.8.1、
NumPy 2.4.4。原始 artifact manifest 没有完整记录这些版本，这是已知 provenance 缺口。

## 模型身份与 Base 修正

正确模型被固定为：

- repo ID：`Qwen/Qwen3.5-0.8B-Base`
- immutable revision：`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- architecture：`Qwen3_5ForConditionalGeneration`
- hidden size：`1024`
- 本地清单：12 个文件、`1,769,913,749` bytes
- model manifest SHA-256：`08da02c07081c28eb9c6605de182924fa4be97d2b3f47d5c37283cab267e2c63`

先前误取的不带 Base checkpoint 没有被删除，也没有被静默覆盖；其不完整 feature cache 没有
complete manifest。正确 Base 的 identity gate 生成了独立 cache identity，benchmark、optimizer、
training、evaluation 和 export 均绑定到该 identity。最终复核重新计算了 12 个 Base 文件的
SHA-256，0 mismatch。

Base checkpoint 没有 chat template。真实样本 smoke 首次暴露该问题后，backbone 增加了显式
`base_multimodal` prompt 路径，使用 vision placeholder 与固定 prompt；该修正发生在完整 cache、
benchmark 和任何真实 policy optimization 之前。

## 数据、清洗与 split

数据身份：

- repo ID：`lerobot/aloha_sim_insertion_human`
- immutable revision：`cc571a3c661df81b566dbfde3d5c1e85fcdf7884`
- 50 episodes，25,000 frames，50 Hz
- task label：`Insert the peg into the socket.`
- observation：top camera + 14-D robot state
- expert action：14-D absolute joint-position target

conservative cleaning 扫描了 25,000/25,000 rows，校验了 revision-scoped cache checksum、episode/
frame 连续性、timestamp、terminal frame、图像 finite/range/shape、state/action 和 instruction。
结果为 `validated_clean`、0 issue、0 row removed。这里的“cleaning”是验收与隔离，不是未经视频
同步重写的行级删除。

固定 split 仅按 episode 进行：train 40、validation 5、hidden test 5；seed 为 `20260809`。
normalization 只使用 train split。特征选择 frame stride 为 5，action chunk length 为 8，最终
缓存样本数为 train 3,960、validation 495、test 495。

## Rosetta Action Contract

Action Contract SHA-256：`8c3263011173d2d978ccef8fccdaafccf3b2a8690b47798a5b55fa69b5c40a9a`。

- type/dimension：continuous，14-D
- semantics：absolute joint-position targets
- space/control：joint-space position control
- units：12 个 arm joint 为 radian；2 个 gripper 为 normalized `[0, 1]`
- ordering：left 7 dimensions followed by right 7 dimensions
- gripper：`0=closed, 1=open`；simulation adapter 扩展为 opposed finger actuator pair
- control frequency：50 Hz
- timestamp：observation at `t` predicts action `t` through `t+7`
- chunk execution：receding-horizon first action
- limit handling：Contract projection/clipping；source gripper overshoot tolerance 为 `0.20`

Gate 2 和 feature cache 都记录了 source action clipping。训练 target 不是未经说明地改变；其
manifest 明确写入 `clip_to_rosetta_contract_v1`。这也意味着结果只适用于当前 Contract 语义，
不能仅凭相同 tensor shape 复用到其他机器人或控制模式。

## 阶段门禁与时间顺序

| 顺序 | 阶段 | 结果 | 可核对证据 |
|---:|---|---|---|
| 1 | 静态 contract/shape tests | passed | test suite |
| 2 | 环境检查、unit tests、dummy/CPU smoke | passed | final QA 记录 |
| 3 | Gate 1 Scripted Action | passed | `gate1-736c5bf0ea1c.json` |
| 4 | Gate 2 Dataset Replay | passed | `gate2-b9c8eaf28601.json` |
| 5 | Base 真实样本 forward smoke | passed | Qwen/data tests |
| 6 | Frozen feature cache | complete | identity `02532ae2...` |
| 7 | **Pre-training benchmark** | complete | SHA `0e19e51d...` |
| 8 | One-step real optimizer smoke | passed | prerequisite 内含 benchmark SHA |
| 9 | Fixed small-set overfit | passed | 32 samples / 300 steps |
| 10 | Full train epoch 1–2 | intentionally stopped | `epoch-002.pt` |
| 11 | Explicit resume epoch 3–39 | complete | best epoch 31，early-stop epoch 39 |
| 12 | Export + exact reload | verified | max absolute diff `0.0` |
| 13 | Validation evaluation | complete | hidden test unopened |
| 14 | Hidden test evaluation | complete | 首次打开 test split |
| 15 | Gate 3 Small Policy Rollout | safety criteria passed | 20-step closed loop |
| 16 | Gate 4 Development Evaluation | safety criteria passed; task failed | 5 × 100 steps |

训练前 benchmark report 的 stage 为 `pre_training`、`hidden_test_loaded=false`。optimizer smoke
report 将 benchmark、Gate 1 和 Gate 2 的 SHA-256 写为不可变 prerequisites。因此本次真实 policy
optimizer 的第一步发生在 benchmark 之后。此前 unit test 的 dummy optimizer 不使用本次完整
Base feature cache，不构成真实实验 optimization。

## Benchmark

Benchmark 只评估 validation 495 samples，没有读取 hidden test。

| 方法 | Action MAE | RMSE | 说明 |
|---|---:|---:|---|
| Train-action mean | 0.1773325 | 0.2338003 | 预先声明的 primary baseline |
| Current-state persistence | 0.0570195 | 0.1394681 | 更强的运动连续性参考 |
| Deterministic untrained policy | 0.1821213 | 0.2403938 | 未训练 Rosetta head |

Benchmark report SHA-256：`0e19e51d7e73ac6f311fe907953c518dce7f9edf1e0a6d290c453615f5e25404`。

## Frozen Feature Cache

Feature cache identity：`02532ae2b512d3e78055811789e6fd3bc14e56f0df5f8b053ed87f9600414ead`。

缓存身份绑定了 Base model/revision、processor、`base_multimodal` prompt、100,352 pixel preprocessing、
final hidden state、attention-masked mean pooling、float16 storage、dataset revision、camera/field mapping、
fixed split、Action Contract 和代码工作区快照。共 50 个 episode shard；构建耗时 `7,042.0 s`。
最终复核重新计算全部 50 个 shard SHA-256，0 mismatch。

这是 frozen experiment 的不可变 pooled cache，不能用于 LoRA/full fine-tuning，也不能复用到 9B。

## Optimizer、overfit、训练与 resume

One-step optimizer smoke：

- normalized Smooth L1：`1.0632218 -> 0.9764350`
- prediction shape：`[1, 8, 14]`
- Action Head、Fusion、State Encoder 均有 finite non-zero gradient
- trainable parameters：493,168

Fixed small-set overfit：32 samples、300 steps，loss `0.5600701 -> 0.0005194`，ratio
`0.0009273`，低于预先设定的 `0.20` 上限。

完整训练配置：batch size 64、learning rate `3e-4`、weight decay `1e-4`、gradient clip 1.0、
maximum 40 epochs、early-stopping patience 8、seed `20260809`。第一次运行在 epoch 2 后按计划停止，
随后显式从 `epoch-002.pt` resume；最终完成 epoch 39 并 early stop。最佳 epoch 为 31。

39 个 checkpoint 的 SHA-256 全部重新计算，0 mismatch。最佳 checkpoint SHA-256：
`a8f60c31a37fec5e9e4d29c361771b1e98841e2906909f5466b45dbc8c8f0727`。

## 离线评估

| Split / 方法 | Samples | Action MAE | RMSE | Raw invalid action rate | Raw projection element rate | Final invalid / limit rate |
|---|---:|---:|---:|---:|---:|---:|
| Validation train-action mean | 495 | 0.1773325 | 0.2338003 | 0.0% | 0.0% | 0.0% |
| Validation current-state persistence | 495 | 0.0570195 | 0.1394681 | 5.25% | 0.375% | 0.0% |
| Best trained validation | 495 | 0.0230362 | 0.0396249 | 33.74% | 1.403% | 0.0% |
| Trained hidden test | 495 | 0.0291032 | 0.0578747 | 22.63% | 0.945% | 0.0% |

最佳 validation MAE 相对声明的 train-action-mean baseline 降低 `87.01%`，相对更强的
current-state-persistence reference 降低 `59.60%`。hidden test MAE 比 best validation 高
`26.34%`。

需要强调：最终 invalid/limit rate 为 0 是 Action Contract projection 后的结果。validation 中
33.74%、hidden test 中 22.63% 的原始 action vector 至少包含一个需投影的元素；这不是可以忽略
的实现细节。当前 policy 明显依赖 projection，后续实验应优化 raw action 合法性，而不是只报告
执行后的 0 违规。

## Artifact export 与 reload

artifact ID：`m2-qwen08b-frozen-001-base-dc7cdfe2`。

导出只包含 Rosetta action components、配置、normalization、Action Contract 和 Model Card，
不重复包含未修改的 Base 权重。artifact manifest status 为 `verified`；从导出 artifact 独立加载
2 个样本的 maximum absolute difference 为 `0.0`。

artifact manifest SHA-256：`9b8955e1902d191560732222ffd261d0595b7ce9aaf1d2cd9e197e0430b08369`。
5 个导出文件的 SHA-256 最终复核均为 0 mismatch。没有上传到 Hugging Face 或其他外部平台。

## MuJoCo Gate 结果

### Gate 1 — Scripted Action Smoke

14/14 dimensions 通过小幅确定性 action 检查；左右臂镜像检查通过；invalid action 被拒绝；
out-of-range action 被 Contract clipping。没有 NaN/Inf 或明显数值发散。

### Gate 2 — Dataset Action Replay

episode 0 重放 100 steps：direction agreement `0.859649`，mean/max target tracking MAE
`0.019370 / 0.047819`，timestamp 最大 step error 约 `1.0e-7`。90/1,400 source elements 被裁剪，
全部来自左右 gripper，且在明确的 source overshoot tolerance 内。

### Gate 3 — Small Policy Rollout

20-step receding-horizon closed loop 可运行；artifact reload verified；invalid action、raw/executed limit
violation、joint-limit violation 均为 0。任务未成功，maximum reward 为 0，unexpected collision 为
80。mean policy inference 为 `2.225 s`，mean simulation step 为 `0.143 s`。这是修正碰撞分类前的
历史 report；修正后相同 seed/20-step rollout 的 reset unexpected collision 与累计 unexpected
collision 均为 0，其他行为指标保持一致量级。

### Gate 4 — Development Task Evaluation

历史 100-step run 用于发现验收和碰撞指标缺陷；审计修正后，使用相同 5 个 deterministic seeds、
环境完整上限 300 steps，共 1,500 steps 重新评估：

| 指标 | 结果 |
|---|---:|
| Task success rate | 0.0% |
| Successful episodes | 0 / 5 |
| Maximum reward | 0.0 |
| Mean rollout length | 300 |
| Invalid action rate | 0.0% |
| Raw / executed limit violation rate | 0.01905% / 0.0% |
| Joint-limit violations | 0 |
| Reset unexpected collisions | 0 / episode |
| Corrected unexpected collisions | 0 |
| Mean action smoothness L2 | 0.0186515 |
| Mean policy inference | 1.4495 s / step |
| Mean simulation step | 0.1525 s / step |

旧报告中的 `1,860` 是修正前的历史计数，不能用于判断碰撞严重度。独立 no-op 诊断发现 reset 姿态
每帧稳定产生左右夹爪内部两指各 4 个 contact point，旧分类器把这 8 个正常同臂接触全部误报
为异常。任务失败只由 5/5 `success=false` 和所有 trajectory maximum reward 为 0 支持；不能用
`1,860` 作为附加严重度证据。修正后的分类只排除同一 arm namespace 内的两指接触，跨臂夹爪
接触及其他 robot/table/object 异常接触仍计数，并记录 canonical pair 分布。完整复验没有观察到
修正定义下的异常 pair；失败来自 task success 和 raw-action contract 两项。

此外，50 Hz contract 的控制周期是 20 ms，而本地 CPU mean policy inference 约 1.46 s；当前闭环
仅用于功能验证，不具备实时控制性能。

## 失败、介入与例外记录

1. 用户指出最初误取了非 Base checkpoint。该错误模型及其不完整缓存被保留隔离，没有删除、
   覆盖或用于本实验；随后在明确授权下获取了正确 Base revision。
2. Base 模型没有 chat template。首次真实样本 smoke 在完整 cache 前失败；加入
   `base_multimodal` prompt contract 后复验通过。
3. 第一次 validation eval 命令漏写了 experiment 级 artifact 目录，容器在读取 artifact 前失败；
   没有打开 validation/test dataset。修正挂载内路径后完成 evaluation。
4. 首次 full training 在 epoch 2 后是计划性停止，用于证明显式 resume；后续从该 checkpoint
   恢复，而不是重头训练并冒充 resume。
5. 本机无合适 GPU，整个 0.8B frozen pipeline 使用受限 CPU 容器。没有通过扩大 swap、内存或
   转向 9B 掩盖接口或实验问题。

## 限制与不确定性

- 历史 Gate 4 的 pass 条件不包含 task success 或 collision threshold；旧 JSON `passed` 不能被
  解读为 task-capability pass。当前严格协议已修复该语义并如实返回 `failed`。
- 只有 5 个 hidden-test episode，置信区间和跨 trajectory 泛化证据有限。
- 数据只有一个 instruction 和 top camera，不能证明语言、视角或任务泛化。
- frame stride 5 的离线特征采样与 50 Hz 在线 rollout 存在 observation distribution 差异。
- pooled final hidden state 会丢失 token-level/spatial detail；Frozen backbone 也限制任务适配。
- 原始动作较高比例依赖 Contract projection；执行后安全指标会掩盖 raw policy 质量。
- collision 仍是启发式 contact-point 计数；虽然已保存 canonical pair 并校准静止基线，但尚未按
  任务阶段或严重程度拆分。
- artifact reload 只做了 2 个样本的精确一致性检查，足够证明序列化合同，但不是广覆盖回归。
- `checkpoint_every_epochs: 5` 在训练快照中未被运行时代码读取；实际每个 epoch 都保存 checkpoint。
  这是配置与行为的不一致，不能用配置字段解释现有 39 个 checkpoint。
- cache/artifact provenance 记录分支为 `m2-qwen08b-frozen-001`，最终验收时活动分支为
  `codex/m2-qwen08b-frozen-001`。旧 manifest 的 107-file workspace tree hash 仍准确；分支值差异
  被保留披露，现有证据不足以把原因唯一归结为 fallback parser。
- 没有第二次从头、同 seed 的完整训练对比；现有证据证明配置、resume、重载与命令可复核，
  不能声称训练曲线或最终权重已被独立复现。
- artifact 内的原始 `MODEL_CARD.md` 生成早于 hidden test 和 Gate 3/4，未披露 hidden-test、raw
  invalid rate、0% task success 及碰撞指标修正；最终审计信息由本报告和
  [Model Card evaluation addendum](m2-qwen08b-frozen-001-model-card-addendum.md) 补充。
- 工作区是 dirty snapshot；revision + workspace tree hash 提供追踪，但正式发布仍应走受保护分支、
  AutoReview 和 PR，不得直接推 main/master。

运行包版本补充：ML image 包含 Transformers 5.14.1、PEFT 0.20.0、LeRobot 0.6.1、PyArrow
25.0.0、NumPy 2.2.6；simulation image 包含 Python 3.11.2、Gym-ALOHA 0.1.4、Gymnasium 1.3.0、
MuJoCo 3.8.1、NumPy 2.4.4。这些版本是审计补充，原 artifact manifest 未完整记录，属于 provenance
缺口。

## 最终复核

2026-08-09 在同一 Docker 隔离边界内复验：

- `scripts/check_env.py`：Linux / Python 3.13.5 / PyTorch 2.11 CPU；通过
- `ruff check .`：通过
- 非 data tests：63 passed、1 CUDA skipped、5 deselected（含碰撞分类与严格 Gate 4 protocol 回归）
- M2 real-data optimizer smoke test：1 passed
- Base model offline inspect：validated，12 files
- feature cache inspect：complete，3,960/495/495 samples
- SHA-256：12 model files、50 feature shards、39 checkpoints、5 artifact files，全部 0 mismatch

只读 pytest cache warning 来自容器 root filesystem 为 read-only，不影响测试结果。

## 可复现命令记录

以下命令从 repository root 的 WSL Bash 运行。下载命令仍需单独的用户授权；再次运行写入命令前
应使用新的 experiment/output identity，避免覆盖本报告绑定的不可变证据。

```bash
scripts/run_m2_container.sh build

scripts/run_m2_container.sh model \
  python scripts/prepare_model.py prepare \
  --config configs/models/qwen35_08b_base.yaml
scripts/run_m2_container.sh model-inspect \
  python scripts/prepare_model.py inspect \
  --config configs/models/qwen35_08b_base.yaml

scripts/run_m2_container.sh data \
  python scripts/prepare_data.py prepare \
  --config configs/data/aloha_sim_insertion_m2.yaml

export ROSETTA_MODEL_ROOT="$PWD/models/Qwen--Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

scripts/run_m2_container.sh sim python scripts/sim_gate.py scripted
scripts/run_m2_container.sh sim python scripts/sim_gate.py replay

scripts/run_m2_container.sh ml \
  python scripts/cache_features.py smoke \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml
scripts/run_m2_container.sh ml \
  python scripts/cache_features.py build \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml

scripts/run_m2_container.sh ml \
  python scripts/benchmark.py \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json

scripts/run_m2_container.sh ml \
  python scripts/train_m2.py smoke \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json
scripts/run_m2_container.sh ml \
  python scripts/train_m2.py overfit \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json

scripts/run_m2_container.sh ml \
  python scripts/train_m2.py train \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json \
  --run-id m2-qwen08b-frozen-001 \
  --stop-after-epoch 2
scripts/run_m2_container.sh ml \
  python scripts/train_m2.py train \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json \
  --run-id m2-qwen08b-frozen-001 \
  --resume checkpoints/m2-qwen08b-frozen-001/m2-qwen08b-frozen-001/epoch-002.pt

scripts/run_m2_container.sh ml \
  python scripts/export.py \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json \
  --training-manifest runs/m2-qwen08b-frozen-001/training/m2-qwen08b-frozen-001/training_manifest.json \
  --checkpoint checkpoints/m2-qwen08b-frozen-001/m2-qwen08b-frozen-001/epoch-031.pt \
  --artifact-id m2-qwen08b-frozen-001-base-dc7cdfe2

scripts/run_m2_container.sh ml \
  python scripts/eval.py \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json \
  --artifact artifacts/m2-qwen08b-frozen-001/m2-qwen08b-frozen-001-base-dc7cdfe2 \
  --split validation
scripts/run_m2_container.sh ml \
  python scripts/eval.py \
  --config configs/experiments/m2_qwen08b_frozen_001.yaml \
  --feature-manifest feature_cache/m2-qwen08b-frozen-001/02532ae2b512d3e7/manifest.json \
  --artifact artifacts/m2-qwen08b-frozen-001/m2-qwen08b-frozen-001-base-dc7cdfe2 \
  --split test

scripts/run_m2_container.sh sim \
  python scripts/sim_gate.py small-rollout \
  --artifact artifacts/m2-qwen08b-frozen-001/m2-qwen08b-frozen-001-base-dc7cdfe2
scripts/run_m2_container.sh sim \
  python scripts/sim_gate.py task-eval \
  --artifact artifacts/m2-qwen08b-frozen-001/m2-qwen08b-frozen-001-base-dc7cdfe2 \
  --seeds 1000 1001 1002 1003 1004 \
  --maximum-steps 300 \
  --minimum-task-success-rate 0.2 \
  --maximum-unexpected-collisions 0
```

## 证据索引

| 证据 | SHA-256 |
|---|---|
| Model manifest | `08da02c07081c28eb9c6605de182924fa4be97d2b3f47d5c37283cab267e2c63` |
| Dataset manifest | `f3c03f7f60db66496d10e12f081c2bd9949a2f9f771148c7f159263dcea98140` |
| Cleaning report | `fe493a1216892367b2dfb2c023065f6079b7cc451144521685d566a243d17a94` |
| Action Contract | `8c3263011173d2d978ccef8fccdaafccf3b2a8690b47798a5b55fa69b5c40a9a` |
| Feature manifest | `8ba0f3a0bc9b8271f9beb754173caf9e5b3a4de631e3657cae2d0556af0130c9` |
| Benchmark report | `0e19e51d7e73ac6f311fe907953c518dce7f9edf1e0a6d290c453615f5e25404` |
| Optimizer smoke | `e933f66631af6bf4b1a84d4fb452c6d4d1764200c997229962cb041f83680922` |
| Overfit report | `26eb06acceca32f7ab0fab8b83f551da7bf59cb7c43ed2e9b8ed02391ff48057` |
| Training manifest | `5b4ea646414e06e028347febf7f922394f3dfd98062e53fd1fc01f31ec20fc44` |
| Best checkpoint | `a8f60c31a37fec5e9e4d29c361771b1e98841e2906909f5466b45dbc8c8f0727` |
| Artifact manifest | `9b8955e1902d191560732222ffd261d0595b7ce9aaf1d2cd9e197e0430b08369` |
| Validation evaluation | `56d39427ac3e7db45767fe2377f9a82ee49def83df7fd7561a66086719baf78d` |
| Hidden-test evaluation | `4916a9a1170edb8c1e015658210fca4d04f2cf294f791bbe11e736218f1ece7f` |
| Gate 1 | `d71e87b219ada503e66ad084fc465885a202816b296ed87ca37cb673ecc2479d` |
| Gate 2 | `6927a0601f68c0dd2ed1ac7bce6c1808342a102215c7a40e28d0851db626a713` |
| Gate 3 historical | `fde3c8afdf7a2951c4d0cafa06a3497ae29b249294e9943ca5711a379a282eca` |
| Gate 3 collision-corrected | `c90634d85def7038f55d636b5e0d904ae205b53aef7bb4fe7af65b698054647e` |
| Gate 4 historical | `9442c59fb2e661550257a484faf6470029d18d0867d09648df28e44a547c1e66` |
| Gate 4 strict 5 × 300 | `b5d2313ae4e0f9fb619d1fb21b4c61c9f574eebe3384366ac2fdedf448191733` |

## 建议的下一轮假设

在保持 0.8B、dataset split、Action Contract、evaluation protocol 和 seeds 不变的前提下，先定位
任务失败层，而不是扩大模型：

1. 对 expert trajectory 在同一初始状态下做完整 task replay，按 body-pair/phase 拆分 collision，
   校准 reward/success detector，确认仿真任务本身可由当前 Contract 完成。
2. 画出 dataset action 与 policy raw/projected action 的 per-dimension 分布，优先解决 gripper 与
   projection 依赖。
3. 做 teacher-forced one-step、short-horizon rollout 和 closed-loop divergence 的分层诊断，确认
   失败来自 representation、action chunking、state feedback 还是 covariate shift。
4. 只有修复后在固定 Gate 4 protocol 上获得非零且可重复的 task success，再讨论 LoRA 或 9B。

## 进一步问题

- expert replay 在与 Gate 4 相同的 5 个 seed/初始状态上能达到多少 success，碰撞基线是多少？
- 任务成功是否依赖未进入 top-camera pooled representation 的空间细节？
- raw action projection 的主要来源是 gripper、joint limit，还是 normalization/head calibration？
- 1.46 s CPU inference 是否会改变仿真 timing，还是 simulator 使用确定性固定步长而不受 wall time 影响？
- 下一轮应先比较 Frozen spatial/token pooling，还是在相同 protocol 上尝试小规模 LoRA？

## 独立审计

独立只读审计确认：正确 Base 身份、50 个 feature shard、39 个 checkpoint、训练前 benchmark
时序、optimizer state 连续 resume、hidden-test 打开时序、epoch-31 artifact tensor 一致性及全部
SHA-256 证据链成立；没有 commit、push、main/master 操作或 AutoReview 绕过。

独立审计不接受旧 Gate 4 `status=passed` 作为 task-capability pass，理由是 5/5 trajectory
`success=false`、task success 0%、maximum reward 0，且旧 pass condition 完全忽略 success 和
collision。审计还通过 no-op baseline 复现了每帧 8 个同臂夹爪内部接触误报，并提出只排除
“同 arm namespace 的两个 gripper finger”这一最小修复。该修复、canonical pair 记录、Gate 4
安全/任务验收拆分与回归测试已实现；旧 JSON 保留用于前后审计，不被覆盖。

独立审计最终判定：Frozen Base 数据—缓存—benchmark—训练—resume—评估—导出—闭环执行链路
可靠；插销任务能力失败；严格 M2 task gate 尚未通过；不得进入 M3 9B scale-up。
