# M2 controlled comparison handoff — 2026-08-10

## 1. 恢复入口与结论

本文件是 `reports/training/m2-controlled-comparison-handoff-2026-08-09.md` 的追加交接，
不覆盖原文件。原交接 SHA-256 仍为：

`66769ba1ab465c4cf9a083e23b7cf1376d1113de40747db36b5e9b2da0703f68`

截至北京时间 2026-08-09 23:51，本轮工作已主动停止新增实验，距 2026-08-10 09:00
硬截止仍有 9 小时以上缓冲。核心结论：

- v006 Instruct control 已完成 optimizer smoke、overfit、正式训练、显式 resume、export/reload、
  validation-only evaluation 和离线诊断，但按固定 selection rule 被拒绝。
- 为解释 v004 Gate 4 失败，已完成 5/5 validation episode 的双环境逐步轨迹、
  teacher-forced / closed-loop 分解、记录域原图 probe 和 image × state 析因 probe。
- 诊断表明记录域本身已有非零预测误差；从记录域切到对齐仿真时，state shift 对策略输出的
  影响大于 image shift，右夹爪最敏感。时间索引 dataset action 不是 recovery oracle。
- 基于上述证据，仅运行了两个预注册单变量候选：v007 `state_dropout=0.1` 与 v008
  `state_noise_std_normalized=0.05`。二者均完成正式 XPU 训练和显式 resume，但都未达到
  validation selection 门槛，已拒绝。
- v006、v007、v008 均未进行 hidden-test evaluation，均未进入 Gate 3/4。
- **M2 仍未完成**。v004 仍是受控参考，不是通过 Gate 4 的可发布模型。
- 没有下载模型或数据，没有删除或覆盖文件，没有 commit、push 或修改 `main`/`master`。

## 2. 当前 Git、Docker、WSL 与资源边界

- branch：`codex/m2-qwen08b-frozen-001`
- HEAD：`c0aa0bd0490c6655c098ccf73b9bc531fa9d9c96`
- workspace：dirty；保留所有既有用户改动，不得 cleanup/reset
- 交接前 workspace code identity：127 files / `2cbe0a319cb961f51f6a84d95f9984154e25a2efe44f7dc07c0946c6adc5a88d`
- Docker Desktop：client/server `29.6.1`，Linux amd64 daemon 健康
- WSL：`2.7.11.0`；Debian 与 `docker-desktop` 均为 WSL2、保持 Running
- XPU：PyTorch `2.11.0+xpu`，Intel Graphics 1 device；`/dev/dxg` 与 WSL GPU libraries 可用
- 每个容器：`5g` memory、`5g` memory+swap hard limit、`2` CPU、`512` PIDs、
  root/workspace read-only、`--network none`
- 交接时没有 running container，也没有残留训练、仿真或模型加载进程；Docker/WSL 保持开启

本轮实际使用的不可变镜像：

| role | image ID |
|---|---|
| ML CPU | `sha256:c107d41f49b92b1a8a8f0c7848802b860cbbe467eae698253e1a76d648f0e3d6` |
| ML XPU | `sha256:449ef8e059edaae7b88499ab6dd90ee3d376be841258c3a5c6eeac26d31fef74` |
| Sim CPU | `sha256:fcacf7315c6880ba97d11a95af67e4c54c0cc7d9e76af0a93b61bd0520bd9f37` |
| Sim XPU | `sha256:f5227bffe19f3c1e8e6c0e2e0fa9699998a7eac718acde26a9ff10ff157163f9` |

## 3. v006 Instruct control：完成但拒绝

配置：`configs/experiments/m2_qwen08b_frozen_006_instruct_spatial_xpu_control.yaml`

- config SHA-256：`92f11d46822645f748f9b2b7e35bf535925bc75d8155145a0e59834c9517a003`
- frozen cache identity：`7e0bb793c1769cc6ce795d38d375eeeed96c1d27f627e35782633c3fec44f145`
- cache manifest SHA-256：`3ef0ef5d0347faa6705220799b16f9f2ea5091d1a73dfb7ed7198ddd3ac9ee65`
- pre-training benchmark：validation 495 samples；`hidden_test_loaded=false`
- XPU optimizer smoke：passed；fixed sample loss `1.07361 -> 0.967834`
- 32-sample overfit：passed；`0.56307 -> 0.00032186`
- run id：`m2-qwen08b-frozen-006-instruct-xpu-001`
- epoch 5 主动停止后，从 `epoch-005.pt` 显式恢复；最终 epoch 40
- best epoch/checkpoint：40 / `epoch-040.pt`
- checkpoint SHA-256：`0c6fad82681037c9387cc4256c2c1451abd586b9cce4eb6653384271d7c55bf7`
- training manifest SHA-256：`e214dbdaee1f3ff3fca06a2d907312d27ca0aa7a32220d97de1253b258f3221f`
- artifact：`m2-qwen08b-frozen-006-instruct-spatial-xpu-control-instruct-2fc06364`
- artifact manifest SHA-256：`7b879b825bc6955ef83a134a834c843a4e6ea27568d34f25ce6eca7904cace7a`
- export reload：2 samples，maximum absolute difference `0.0`
- independent validation action MAE：`0.0270894933`
- raw action limit violation rate：`0.0195526704`
- evaluation：validation only，495 samples；`hidden_test_opened=false`

与 v004 Base XPU reference 比较：

| metric | v004 reference | v006 | 判定 |
|---|---:|---:|---|
| validation action MAE | `0.0266666878` | `0.0270894933` | 更差 |
| raw limit violation rate | `0.0101912` | `0.0195527` | 近乎翻倍 |
| validation feature-ablation MAE delta | `0.0300418` | `0.0327571` | 有视觉响应，但不足以抵消主指标退化 |
| within-frame feature-shuffle MAE delta | `0.0195899` | `0.0200061` | 有视觉响应，但不足以入选 |
| reset predicted/target pairwise L2 ratio | `0.41955` | `0.39139` | 对 reset 任务差异的响应更弱 |

因此 v006 已拒绝。它还是 Instruct auxiliary control，配置明确
`m2_completion_eligible=false`，即使离线指标更好也不能单独满足 Base-only M2 完成条件。

## 4. v004 validation-only 轨迹分歧证据

### 4.1 协议边界

新增轨迹协议只允许配置声明的 validation episodes `[22, 13, 7, 33, 45]`，并在任何
dataset/model/simulator I/O 前拒绝 train、test、重复或未声明 episode。协议还要求：

- validation-only 分片初始图像；undeclared shard/file 在 tensor load 前拒绝；
- dataset revision、manifest、config、artifact、Action Contract、alignment report 全部交叉绑定；
- 两个独立 simulator backend，同 seed reset，state 与所有 camera image 精确对齐；
- smoke 固定 3 步；full 必须完整 episode，并重算、匹配 smoke 三步 canonical prefix；
- 每步只执行 action chunk 的首动作并重新观察；strict finite JSON、create-only report；
- 固定 state divergence thresholds：`0.01, 0.025, 0.05, 0.1`；
- dataset expert 明确标注 `time_indexed_expert_reference`、`state_conditioned=false`、
  `recovery_oracle=false`。

validation initial-image manifest SHA-256：
`c0bcb7d1add1ded09a627b06cb040798ba9e1508c9edc98ed58f8890afbc071d`

alignment report SHA-256：
`29f6de0cfac5f7871fbabc4c729c3e93b49c0c638feab7736b4bbe8cc443e2f3`

selected seeds：episode 22/13/7/33/45 -> `29/20/15/39/5`；所有 pooled 4x4 MAE
均小于固定 `0.005`。

### 4.2 5/5 full trace 汇总

五个 episode 均完成 3-step smoke 与 500-step full，prefix 全匹配，共 2500 paired steps：

- median step-0 action L2：`0.103721`
- median step-0 state L2：`0.049895`
- median step-1 state L2：`0.092871`
- median state-MAE first crossing steps for `0.01/0.025/0.05/0.1`：`1/10/19/29`
- median maximum state MAE：`0.275447`
- median final state MAE：`0.266501`
- policy non-zero reward：5 / 2500 steps，仅 episode 45 出现
- policy raw clipping：573 steps / 581 elements；所有 episode 的首次 clipping 都晚于初始分歧
- policy unexpected collision：0；expert unexpected collision：1

结论：早期分歧在 5/5 validation episodes 系统性复现，不能由后期 clipping、collision 或
joint-limit event 解释。但分岔后的 dataset 时间索引 action 不是基于 policy state 的恢复动作，
因此 full trace 不能被表述为相对 recovery oracle 的动作误差。

### 4.3 三流、记录域和析因诊断

teacher-forced decomposition report SHA-256：
`52b4b1f3443c388ae911038d30d4d556196675647fd2c0e4f797b13302f4f5f8`

- 5 episodes × 3 steps；step-0 相同输入最大差异 `0.0`
- policy-on-expert-stream vs dataset action median L2：
  `[0.103721, 0.085434, 0.096785]`
- closed-loop vs policy-on-expert-stream median L2：
  `[0.0, 0.027845, 0.032292]`

记录域原图 probe report SHA-256：
`b92f5363f4cec1702b219dbfe83a3826079bf9b77c441f4fc4edb0d4bfb54a82`

- exact validation episodes；frames `[0, 1, 2, 5, 10]`；25 samples；test 未打开
- frame-0 recorded prediction vs dataset action median L2：`0.0746575`
- frame-0 median MAE：`0.0137591`

image × state factorial report SHA-256：
`a21694285e59a331cdd9e4140f2c6efa81d9fc543be774af6044b83d123e648a`

| median response L2 | value |
|---|---:|
| state swap at recorded image | `0.0506469` |
| image swap at recorded state | `0.0302188` |
| image × state interaction | `0.00913417` |
| joint recorded -> sim swap | `0.0605821` |
| recorded-domain error | `0.0746575` |
| simulator-domain error | `0.1037211` |

这说明 v004 并非只在闭环后才出错：记录域 step 0 已有监督/表示误差；对齐 simulator reset 的
小 state shift 又进一步改变输出，其影响大于 image shift，且右夹爪最敏感。该结果支持继续研究
train-only state-domain robustness，但不证明单靠 state regularization 就能解决任务。

## 5. v007：state dropout 候选，拒绝

配置：`configs/experiments/m2_qwen08b_frozen_007_state_dropout_xpu.yaml`

- 唯一实验轴：`action_expert.state_dropout: 0.0 -> 0.1`
- config SHA-256：`57d499d9665694531bf30a52fea7a9144141512a988acceddb5e657546bae7f4`
- derived cache identity：`020a3182248a12e732eecc73e18f534e248ddd8b7f834333978157b8c9ab8839`
- cache manifest SHA-256：`66607b5ed258eb6621e26084d1975c229343176909a1ed9c67f76ca108c10abc`
- Gate 1/2：passed；Gate 2 fresh replay 294 steps，task success
- benchmark：validation only；`hidden_test_loaded=false`
- optimizer smoke：passed；`1.07750 -> 0.95381`
- 32-sample overfit：passed；loss ratio `0.000692`
- epoch 5 主动停止，随后显式 resume
- completed epoch：23，early stopped
- best epoch/checkpoint：15 / `epoch-015.pt`
- best checkpoint SHA-256：`f3144cf92c8831189689ccad4931f04cac2ec15d215f418a258dc33f90e11990`
- best validation MAE：`0.0301873740`
- training manifest SHA-256：`dda9aa4a0ea28b158557e70bdee973d1a3f0c95c9b0d362cda594e233e5fc0e5`

预注册上限为 `0.0271666878`；v007 超出 `0.0030206862`，因此立即拒绝。没有 export、
independent eval、factorial、hidden test 或 Gate 3/4。结论是隐藏层 dropout 过于宽泛，明显损害
记录域 validation 拟合，不应继续做 `0.1 -> 0.05` 的 validation 参数扫网格。

## 6. v008：train-state jitter 候选，拒绝

配置：`configs/experiments/m2_qwen08b_frozen_008_state_jitter_xpu.yaml`

新增训练能力只在 training input 上对 normalized robot state 加高斯微扰；label、validation、
evaluation 和 simulator input 保持干净。XPU RNG 写入 checkpoint 并在 resume 时恢复。

- 唯一实验轴：`training.state_noise_std_normalized: 0.0 -> 0.05`
- `state_dropout` 恢复为 v004 的 `0.0`
- config SHA-256：`d2bc52d1356b028717b433cf5ba9144ac3815b90fbd70adb5a85c986ee133dc2`
- derived cache identity：`4a25ba5689bcf96426416c9ffa3b4fdd12c66a403524987d3cb3c0426198979c`
- cache manifest SHA-256：`4acc053aadc9ada161809ea7c7d455dbf80ef3682795c36a188d6b68292faa65`
- Gate 1/2：passed；Gate 2 fresh replay 294 steps，task success
- benchmark：validation only；`hidden_test_loaded=false`
- optimizer smoke：passed；`1.07803 -> 0.95780`
- 32-sample noisy-input / clean-eval overfit：passed；clean loss ratio `0.000746`
- epoch 5 主动停止，随后从 SHA
  `c9840f8391e9e5412581abf2b134778ddcf4ee022a7e2e055d3fe54ef95b41c6`
  的 checkpoint 显式 resume
- completed epoch：40
- best epoch/checkpoint：35 / `epoch-035.pt`
- best checkpoint SHA-256：`56bf580dc422c590c79f5d506d9b4c3718b49bb3953e5eb32901e29270a48e7c`
- best validation MAE：`0.0274870396`
- training manifest SHA-256：`275168bf78d6fc612739c9e5dedd08966f5483bfcf904e23592513a9d4519b48`

相对 v004，v008 validation MAE 高 `0.0008203518`；预注册只允许高 `0.0005`，因此它比门槛仍高
`0.0003203518`。候选已拒绝，没有 export、independent eval、factorial、hidden test 或
Gate 3/4。不得因为“接近”而事后放宽门槛。

## 7. 当前 selection 表

| experiment | changed axis | best/independent validation MAE | selection |
|---|---|---:|---|
| v004 | Base spatial XPU reference | `0.0266666878` | reference；Gate 4 failed |
| v006 | Instruct checkpoint + native prompt | `0.0270894933` | rejected |
| v007 | state encoder dropout `0.1` | `0.0301873740` | rejected before export |
| v008 | train-state jitter std `0.05` | `0.0274870396` | rejected before export |

没有 validation-selected candidate，因此 test split 继续保留。不得为了选模型打开 test。

## 8. 本轮代码与测试变化

增量变化包括：

- `scripts/diagnose_m2.py`：validation-only 分片初始图像、严格 scope/identity、记录域与诊断输入。
- `scripts/sim_gate.py`：双环境 trajectory divergence、teacher-forced decomposition、
  recorded-domain / factorial probes、artifact/contract/alignment 强绑定、strict finite/create-only；
  Gate 2 可从已通过且身份匹配的 Gate 2 report 复用单 episode 对齐结果。
- `scripts/train_m2.py`：训练态 normalized-state jitter；默认值 `0.0` 保持旧实验行为；
  XPU RNG checkpoint/resume 继续生效。
- `tests/test_trajectory_divergence_trace.py`：轨迹协议、I/O 前 hidden-test 拒绝、双 backend、
  smoke/full prefix、nonfinite、create-only 等回归测试。
- `tests/test_m2_diagnostics.py`、`tests/test_sim_gate_protocol.py`、
  `tests/test_m2_training_protocol.py`、`tests/test_experiment_config.py`：对应合同回归测试。
- 新增 v007/v008 预注册实验配置。

所有诊断 report 都绑定各自生成时的 `evaluation_code` identity；后续新增诊断或训练路径没有
覆盖旧 report。不要把不同 code identity 的 immutable report 改写成同一次运行。

## 9. 最终 QA

- `ruff check .`：passed
- ML CPU full pytest：`139 passed, 4 skipped`
  - 1 skip：CUDA 不可用
  - 3 skip：该 image 中 real-data field-mapping smoke 的既定 skip 条件
- Sim CPU 专项：`41 passed`
- `git diff --check`：passed
- v006 / v007 / v008 checkpoint manifests：分别 21 / 16 / 22 个 checkpoint，重新计算 SHA，
  mismatch 均为 0
- v006 exported artifact：5 files，重新计算 SHA，mismatch 0；reload verified，max diff `0.0`
- pytest 唯一 warning：只读 workspace 不能写 `.pytest_cache`；不影响测试结果
- 无 running container；无残留 ML/sim 进程

## 10. 关键 SHA-256 索引

| artifact | SHA-256 |
|---|---|
| v006 benchmark | `f3573ef98f2cf8a541f7f474a304fddff4f3392cf533ebb4b922027eb3c4281c` |
| v006 optimizer smoke | `61f44a20855a4ef3372be4bf358d4abd0c25fb030cc71fed9693fbdfbfbb7bc9` |
| v006 overfit | `0de1fb1e2132d42083485350498a4bed1a9213ae1297e75c62a743e8b61b9ce0` |
| v006 validation evaluation | `5b98fdbe229bce125e7324b4c814a50b3b0db2f1dfb316636021f28f25aeeaf2` |
| v006 cached-policy diagnostic | `ebde322dc1a2509183126e657e1209140f3b6da5856bc809011aca11e1940314` |
| v006 modality audit | `c224759e6c2075f6fdf62463171059b4e6d1da06a560b7367fb4f04b7beae85e` |
| v007 benchmark | `68290bf84e96c71f1a0a1de845dc46090e2535927dea1e889fa6f1bad369d4b8` |
| v007 optimizer smoke | `de54e6cf96409c3cc7b47c024586ee4542dd8134f0bc83614c0e58784f89481d` |
| v007 overfit | `e1032f1c2f39294a438a091ffa16304cac5fc7726751b8206947387230a573e1` |
| v008 benchmark | `58fdcc2adff2075e5142aabd02f00feb47d229a380762a2c03679eee264469e1` |
| v008 optimizer smoke | `8d9732cf8f4bd5fe77177a9a0a475d984cd41205374846fb95703431c91512b8` |
| v008 overfit | `0f8f4901a058f96c007d7994bbaad6869a0315f8a23feaa153deadadf308d814` |

## 11. 下一轮建议与禁止事项

下一轮不要继续在同一 validation split 上扫 dropout 或 jitter scale。更有信息量的方向是：

1. 先定义 train-only 的 simulator-domain pairing / state-calibration 数据生成协议，避免用 validation
   reset offset 直接训练；仍需保持 hidden test 关闭。
2. 区分表示/监督误差与 domain shift：记录域 frame 0 已有 L2 `0.07466`，不能只修 simulator。
3. 若引入第三条 teacher-forced training signal，必须明确它不是 recovery oracle；不要把分岔后的
   时间索引 action 当成 state-conditioned recovery label。
4. 优先研究 object pose / task phase 可辨识性和更贴近控制频率的视觉输入；当前 frozen stride-5
   cache 与 50 Hz online control 仍有明显分布差异。
5. 新候选必须重新预注册单一 changed axis、validation selection rule、raw-action 安全门槛和
   simulator diagnostic 门槛；只有全部通过后才能 export、Gate 3/4，之后才讨论一次性 test audit。

禁止事项：

- 不要把 v004、v006、v007 或 v008 标记为 M2 complete。
- 不要运行 v006/v007/v008 Gate 3/4；它们已在 selection gate 前被拒绝。
- 不要打开 hidden test 进行候选选择。
- 不要覆盖或删除现有 cache、checkpoint、artifact、report 或用户改动。
- 不要下载新模型/数据、build/pull/install，除非获得用户新的明确授权。
- 不要 commit/push，除非用户明确要求；不得绕过 AutoReview。

新会话从本文件、`AGENTS.md` 和原 2026-08-09 handoff 开始，只读核验 SHA、branch/HEAD、
Docker/WSL 和无残留进程后，再定义新的 train-only 假设即可。

## 12. 07:25 增量执行：v009–v015

本节是在同一北京时间 2026-08-10 任务窗口内追加的最终封口。所有真实 ML、数据和仿真命令
都在 WSL 的固定 Docker image 中运行；未 build、pull、install 或下载模型/数据。Docker 资源
继续固定为 5 GiB、2 CPU、512 PIDs，网络保持关闭。正式训练设备为 Intel XPU；每个候选均先
通过配置/合同测试、Gate 1、Gate 2、XPU optimizer smoke 和固定 32 样本 overfit，再进行
epoch 5 主动停止与显式 resume。没有改选非预注册 checkpoint，也没有事后放宽阈值。

### 12.1 统一 selection 结果

| candidate | 唯一变更轴 | best epoch | full MAE | first MAE | raw-limit rate | 结论 |
|---|---|---:|---:|---:|---:|---|
| v009 | first-action loss weight `0 -> 1` | 31 | `0.0263042320` | `0.0222968124` | `0.0103896102` | first/raw 门失败；拒绝 |
| v010 | train-only aligned replay state pairing | 28 | `0.0272446983` | `0.0240551103` | `0.0193362199` | offline 门失败；拒绝 |
| v011 | maximum epochs `40 -> 80` | 57 | `0.0262359250` | `0.0218611266` | `0.0065115439` | strict full 门高 `0.0000692372`；拒绝 |
| v012 | visible anchor stride `5 -> 2` | 20 | `0.0254592653` | `0.0215674583` | `0.0088128978` | offline 通过；online domain 门失败 |
| v013 | v012 + first-action loss weight `0 -> 1` | 25 | `0.0245188251` | `0.0195102599` | `0.0144158471` | raw 门失败；拒绝 |
| v014 | v012 + early phase `[0,50)` first-action loss | 34 | `0.0251701716` | `0.0223447178` | `0.0069042798` | first 门失败；拒绝 |
| v015 | v012 fusion width `256 -> 512` | 39 | `0.0244941190` | `0.0200049169` | `0.0103383455` | raw 门高 `0.0001471480`；拒绝 |

v009 使用此前 legacy cache，manifest 中仍物化了 495 个 test samples；本轮没有把这些 tensor
载入 benchmark、训练或 selection。自 v010 起全部新 cache 都是 visible-only：test samples/shards
均为 0，`hidden_test_loaded=false`。没有用 hidden test 选择候选。

### 12.2 v012：完成 export/reload，但 online selection 失败

v012 直接用本地 Base 0.8B 和 XPU 生成 stride-2 visible cache：train `9880`、validation
`1235`、test `0`。best epoch 20 完成独立 validation、artifact export 和独立 reload；reload
maximum absolute difference 为 `0.0`。artifact manifest：

`artifacts/m2-qwen08b-frozen-012-stride2-xpu/m2-qwen08b-frozen-012-stride2-xpu-base-dc7cdfe2/manifest.json`

manifest SHA-256：`972708dc87a910008f2429bb8d77274c9117b3306cdad8c58785544f6a1f1c0b`

随后按预注册顺序运行 online diagnostic：

| diagnostic | v012 | 上限 | 结论 |
|---|---:|---:|---|
| recorded-domain frame-0 median action L2 | `0.0927170813` | `0.0751575370` | fail |
| factorial state-swap median action L2 | `0.0550202765` | `0.0506468825` | fail |
| factorial simulator-domain median action L2 | `0.1129527465` | `0.1037210822` | fail |

因此 v012 没有进入 Gate 3/4。额外 cache 诊断确认 online recorded frame-0 prediction 与 cache
prediction 按 episode 精确一致，排除了 online/cache 推理漂移；reset action 预测的 episode 间
pairwise L2 mean `0.04365`，明显低于 target 的 `0.11210`，更符合模型在 reset 视觉条件下
输出欠离散，而不是缓存读取错误。

### 12.3 v015 最后一轮完整训练

配置：`configs/experiments/m2_qwen08b_frozen_015_fusion512_xpu.yaml`

- config SHA-256：`86bc6a8723ac96dab9b7ccdaa7fbfcef553349e38566d574a741eff9e29146cf`
- visible cache identity：`56fae4fb3c0328dd1c2e1ce3b5c68fb6a96e35b96a61ab3653551298cd0feb48`
- cache manifest SHA-256：`3f5dc594b8287d315127f482c99a567bdedf821cb7dfe77081142b1e9af6fdbe`
- benchmark / Gate 1 / Gate 2：complete / passed / passed
- Gate 2：294 steps，task success；没有神经 policy 参与
- optimizer smoke：`1.069249 -> 0.953624`，state encoder、fusion、action head 梯度均非零
- 32-sample overfit：`0.652983 -> 0.000173664`，ratio `0.000265956`
- epoch 5 主动停止，随后从 `epoch-005.pt` 显式恢复到 epoch 40
- best checkpoint：`epoch-039.pt`
- best checkpoint SHA-256：`81cb1a275f71574bd278a69608e93e504b90f519d03048e039afa906eda1b10a`
- training manifest SHA-256：`ceac7dc1e52660a960bd342017c9666ea08f074ab27b6506e2b2efab277430fc`

v015 的 full/first 离线误差是本轮最低组合之一，但 raw-limit rate 没有满足固定安全门。按配置
`raw_action_limit_violation_rate_at_most=0.0101911975`，因此没有 export、online probe 或
Gate 3/4；不能因差值小而改选其他 epoch。

### 12.4 cache / training 身份索引

| candidate | cache manifest SHA-256 | training manifest SHA-256 |
|---|---|---|
| v009 | `d03e825ed9b55d109ab4b3cd91d07b0aab1fdd409d9e2eb6bd81901c2a317b43` | `3780e777dff26e2bda766a8c3d78a0216e515aa940113efa9d7fb666c9202869` |
| v010 | `504ede13a5fb42acab8f70b24ec13dce1239b9a98c0a12f17ee82308e93c9b6f` | `86489b0ba6a56c92cd7cc03784ad444cc814e9f2a46b64556dd4ba452899f63b` |
| v011 | `17022227a7f23243b96788b576b804c1bf10c8953cc515922125983157fc81e8` | `4890e12f608e457430fb88c95ac696bff8dbc1bd419e94933eac5ada0d6b7b32` |
| v012 | `ef8a065569aff81e3e7c0da633826b14a015ea55d29fee3f7f038abae9ca5466` | `52d26a6c227eb1bafd08ec6a26bb4b44c98e841b30990de32300342aae0dc0be` |
| v013 | `40caaf416f47a7d350ba8aa0c021ab023c2496d759626bf1efeb79872c12e2b2` | `9a1e3ff8ec25500966925ba5294cbbf896cd41d3dc0ac4becda9c4ef1c623018` |
| v014 | `a7a7c61b1c58c01ca5cdc5bdf90d552b3f707db7ce09c316f5af201f97b0247a` | `ac79efdf6537523e4b003b794d178185216a1873e9fe5d6139ca0d6672030d6f` |
| v015 | `3f5dc594b8287d315127f482c99a567bdedf821cb7dfe77081142b1e9af6fdbe` | `ceac7dc1e52660a960bd342017c9666ea08f074ab27b6506e2b2efab277430fc` |

## 13. 07:25 最终状态与 QA

- `ruff check .`：passed
- WSL full pytest：`191 passed, 4 skipped`
- skipped：1 个 CUDA unavailable；3 个当前 real-data field-mapping 的既定 prepare-data 条件
- `git diff --check`：passed（在 handoff 追加前）
- Docker daemon / WSL / XPU：健康；训练完成后无 running container 或残留 ML/sim 进程
- 未删除或覆盖任何 cache、checkpoint、artifact、report 或用户文件
- 未 commit、未 push；工作区仍保留原有未提交修改

本轮已经训练出多个可复现的 Qwen3.5-0.8B-Base frozen development checkpoint，并完成 v012
的 export/reload 与 online domain diagnosis，但没有任何新候选同时通过完整 selection 和 Gate 3/4。
旧 v004 Gate 4 仍是 0/5 success。因此 **M2 仍未完成**，不能把“完成 0.8B 正式训练”误报为
“已获得通过闭环任务验证的 M2 模型”。下一轮应先基于 train-only 证据解决 reset 视觉到动作的
欠离散和 raw-limit 安全性；不得继续在同一 validation split 上扫 fusion width、loss weight、
early-phase window 或挑选非 best epoch。
