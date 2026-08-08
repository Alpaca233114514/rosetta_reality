# Repository agent guidance

## Repository safety

- Never delete, clear, overwrite, or discard existing files unless the user explicitly approves the exact affected paths.
- Before any deletion, list every affected path, explain the reason and a reversible alternative, and wait for approval. Broad requests such as cleanup, refactoring, or repair are not deletion authorization.
- If an operation might lose data and that cannot be ruled out, treat it as deletion and stop.
- Prefer compatible, incremental edits when an existing file or structure overlaps a task.
- Treat `main` and `master` as permanently protected branches.
- Put integration changes on a feature branch and use the normal pull-request workflow.
- Never force-push, rewrite Git history, bypass branch protection, or weaken required checks.
- Never bypass AutoReview.
- Do not commit or push unless the user explicitly requests it.
- Never commit secrets, access tokens, credentials, private data, or model access keys.

## Path privacy

- Treat the repository root as `.`.
- Refer to repository files and directories only with paths relative to the repository root, such as
  `src/rosetta_reality/`, `data/`, or `.venv-wsl/`.
- Do not place absolute Windows, WSL, UNC, home-directory, username, or machine-specific paths in
  tracked files, documentation, logs, handoffs, or user-facing responses unless the user explicitly
  requests a specific absolute path.

## GPU and environment safety

All local machine-learning operations must run inside WSL. This includes creating
or updating Python environments, installing ML or data dependencies, downloading
or preparing datasets, running PyTorch/LeRobot code, ML unit or integration tests,
smoke training, evaluation, and real experiments. Do not perform these operations
from native Windows Python or PowerShell. Repository editing, Git operations, and
non-ML static checks may still run from Windows.

Never install or upgrade system-level PyTorch, CUDA, ROCm, GPU drivers, or simulators automatically.
Before any real GPU training:

1. Run `python scripts/check_env.py`.
2. Run the unit tests.
3. Run the dummy forward pass.
4. Run a CPU or small-model smoke test.
5. Run a short GPU smoke test.
6. Only then launch a real experiment.

Do not download model weights or datasets unless the user explicitly requests it.

## Architecture

- Keep the vision-language backbone replaceable.
- Keep the dataset layer robot- and embodiment-agnostic.
- Do not leak Qwen-specific behavior into generic VLA policy components.
- Do not hardcode action dimensions, chunk sizes, robot degrees of freedom, devices, or dtypes.
- Avoid speculative abstractions without an immediate use.
- Prefer a small working baseline before advanced research features.
- Imports must not load models, access the network, or mutate the environment.

## 模型工作强制工作流

以下工作流适用于数据准备、真实多模态骨干接入、特征提取、训练、评估和模型适配。
后续模型工作默认必须遵守本节；不得为了更快进入下一阶段而跳过阶段门禁。

主力本地开发机只有 **16 GB 系统内存**，且没有适合大模型训练的独立 GPU。
所有本地机器学习操作仍须在 WSL 内执行。Windows 侧只用于仓库编辑、Git 操作和
不加载模型或数据的静态检查。模型权重、数据集、外部 GPU 计算、真实训练和真机操作
均不得在未获得用户明确授权时自行启动。

### 阶段门禁

- 里程碑是否完成必须由当前工作区中的实际验收结果证明，不能仅根据代码存在、PR 已合并、
  文档描述或历史对话推断。
- 当前 M1 到 M2 的门禁是：在 WSL 中完成环境检查、单元测试、dummy forward / CPU
  optimizer smoke，以及 revision-pinned 真实数据缓存的检查和 `pytest -m data`。需要创建或
  补全数据缓存时，只有在用户明确批准下载后才能运行 `python scripts/prepare_data.py`；
  `python scripts/prepare_data.py inspect` 必须保持只读。
- 模型规模、backbone adaptation、Action Expert 和 dataset 是相互独立的实验轴。不得再把
  Frozen、LoRA 或 Full Fine-tuning 与某个里程碑永久绑定；里程碑表示系统已经获得并验收的
  能力，而不是本次实验使用的训练技术。
- **M2 — Development VLA**：把本地已有的 `Qwen3.5-0.8B-Base` 作为正式的
  development-scale reference model。0.8B 不是一次性 smoke checkpoint；必须先用它完成
  第一代 Rosetta 的数据、训练、验证、checkpoint / resume、evaluation、artifact export 和
  可复现闭环，并通过 MuJoCo 基础控制验证。M2 完成时必须已有一个能在明确 action 语义下
  实际驱动目标仿真机器人的 development checkpoint，而不只是能输出 action tensor。
- 0.8B 的每种新训练能力仍须从极小样本开始，依次验证图像预处理、instruction / prompt、
  hidden state、tensor shape、pooling、梯度、机器人状态融合、动作预测和小数据 overfit，
  然后才能扩大为完整实验。不得为了执行这些步骤自行下载权重或数据。
- Frozen、LoRA 和 Full Fine-tuning 可以分别在 0.8B 上形成受控实验。是否运行某个适配方法
  由实验假设、资源和用户授权决定；如需比较方法，应保持 dataset split、evaluation protocol
  和其他控制变量一致，并优先建立可复现的 Frozen reference。
- **M3 — Scale-up VLA**：只有 0.8B 完整闭环验收通过后，才能把已验证的 pipeline 扩展到
  `Qwen3.5-9B-Base`。9B 必须先做短 GPU smoke，再在用户批准的正式 GPU 环境中运行
  Frozen、LoRA 或 Full Fine-tuning 的 controlled experiment。
- 16 GB 本地机器不得承担 9B 正式训练。9B 在本地只允许做资源边界明确的冻结推理；LoRA
  和 Full Fine-tuning 必须在用户批准且经过资源预算验证的 GPU 环境运行。192 GB 等大显存
  不是跳过 0.8B 闭环、测试门禁、短 GPU smoke 或显存规划的理由。
- 不得把 Frozen Backbone + Action Head Training 描述为 LoRA、full fine-tuning 或
  9B 全参数训练。

### 每次模型工作的执行顺序

每项模型改动或实验必须按以下顺序推进，并在上一步有可检查证据后才进入下一步：

1. 定义一个明确假设、变更范围、数据和模型身份、资源预算、验收指标及停止条件，并分别写明
   backbone scale、adaptation method、Action Expert 和 dataset，避免混淆实验变量。
2. 先完成不加载权重和数据的静态检查、配置检查、shape / contract 单元测试。
3. 在 WSL 中运行 `python scripts/check_env.py`、相关单元测试、dummy forward 和 CPU 或
   小模型 smoke test。具体测试范围应与风险相称；进入真实 GPU 训练前仍须执行本文件前述
   完整 GPU 前置检查顺序。
4. 在正式训练前定义完整的 Rosetta Action Contract 和 Simulation Adapter，并依次通过 M2
   Gate 1 Scripted Action Smoke Test 与 Gate 2 Dataset Action Replay。
5. 使用 0.8B 和极小真实样本验证 Online Backbone、选定 adaptation method 和端到端
   forward / backward 路径。
6. 使用很小的固定样本集验证模型能够 overfit，并排查 split 泄漏、梯度、checkpoint 保存与
   resume、日志和 evaluation 实现。
7. 对 Frozen 实验按本文件的 Feature Cache 流程生成并验证不可变缓存；对 LoRA 或 Full FT
   使用 Online Backbone，不能把预计算冻结特征冒充为 backbone adaptation。
8. 在 0.8B 上完成正式 training / validation、checkpoint / resume、evaluation、artifact
   export 和从导出 artifact 重新加载的全流程，并验证结果可复现。
9. 依次通过 M2 Gate 3 Small Policy Rollout 与 Gate 4 Development Task Evaluation，证明
   checkpoint 能经完整 adapter 链路形成基础 observation-action-observation 控制循环。
10. 0.8B 的训练、artifact 和仿真控制验收全部通过后才进入 scale-up gate。切换到 9B 时保持
   可比的数据、split、指标和 pipeline，
   重新生成与 9B 身份匹配的 Feature Cache 或在线训练输入，并完成短 GPU smoke。
11. 只在短 GPU smoke 通过后运行 9B controlled experiment；不能把首次完整训练留到 9B
   阶段才调试 checkpoint、resume、evaluation、export 或数据泄漏问题。
12. 记录配置、结果、失败、人工介入和下一轮假设，再决定是否扩大数据、模型或计算规模。

两级模型工作主路径为：

```text
Dummy / Contract Tests
        |
        v
M2 Gate 1 Scripted Action + Gate 2 Dataset Replay
        |
        v
0.8B Tiny-sample Smoke
        |
        v
0.8B Small-data Overfit
        |
        v
0.8B Full Train / Validate / Resume / Evaluate / Export
        |
        v
M2 Gate 3 Small Rollout + Gate 4 Task Evaluation
        |
        v
Reproducibility and Scale-up Gate
        |
        v
9B Short GPU Smoke
        |
        v
9B Controlled Experiment
```

任何一步失败时，应修复该层并重新验证，不得通过扩大模型、数据、内存、swap 或计算资源
来掩盖接口、数据或实验设计问题。

### M2 — Development VLA 与仿真控制验证

M2 的目标不是仅让模型成功输出动作张量，而是建立第一条从真实多模态 observation 到仿真
机器人实际运动的完整控制链路。M2 默认使用开发规模 backbone，例如
`Qwen3.5-0.8B-Base`，完成端到端训练、评估、checkpoint、恢复、导出及仿真控制验证。

M2 完成后，应存在一个能够实际驱动 MuJoCo 中目标机器人模型的 Rosetta Reality
development checkpoint。

#### M2 控制链路

```text
Image / Video
+
Language Instruction
+
Robot State
        |
        v
Rosetta Reality Policy
        |
        v
Rosetta Action
        |
        v
Embodiment / Simulation Adapter
        |
        v
MuJoCo Controller / Actuator
        |
        v
Robot Motion
        |
        v
Next Observation
```

模型输出不能仅被视为无语义 tensor。Rosetta Action Contract 必须至少明确记录：

- action 类型和 dimension；
- 每一维的物理含义及 joint / actuator ordering；
- 单位；
- absolute 或 delta 语义；
- position、velocity 或 torque 控制模式；
- joint-space 或 end-effector-space；
- 使用的 reference frame；
- control frequency 和 timestamp alignment；
- action chunk length 及 chunk execution 策略；
- gripper 等离散或连续控制字段的编码与语义；
- joint limit、actuator limit 和输出裁剪规则。

两个 action tensor 即使 shape 完全相同，只要任一物理语义不同，就不得视为兼容。

#### M2 仿真选择

M2 优先使用现有、高质量、已验证的机器人仿真模型。当前 ALOHA 数据路径应优先使用
MuJoCo 中可获得的 ALOHA 模型，而不是在 M2 自行设计新的机械臂。

SolidWorks、CAD-to-URDF 或自定义机器人建模不属于 M2 的必要前置条件。M2 的主要研究变量
应保持在数据、representation、policy、action space、controller adapter 和 simulation
integration，不应转移为机械结构设计。

#### Gate 1 — Scripted Action Smoke Test

首先不使用神经网络。通过人工构造的小幅、确定性 action 验证：

- 正确的机器人、joint 和 actuator 被控制；
- 运动方向与 action magnitude 符合定义和单位；
- left / right arm 不发生索引交换；
- gripper 语义正确；
- joint limit 和 actuator limit 生效；
- 不出现 NaN、Inf、异常瞬移或明显数值发散。

Gate 1 失败时，不得继续 dataset replay、policy rollout 或正式训练。

#### Gate 2 — Dataset Action Replay

将真实训练数据中的 expert action 通过 Rosetta Action Adapter 输入仿真环境，验证 dataset
action semantics、Rosetta internal action schema 与 simulator control semantics 三者一致。

不得仅凭 action dimension 相同判定 dataset 与 simulator 兼容。必须明确检查：

- joint ordering；
- absolute / delta；
- position / velocity / torque；
- meter / millimeter；
- degree / radian；
- control frequency；
- timestamp alignment；
- reference frame；
- gripper encoding。

如果 expert trajectory replay 产生明显不合理运动，不得开始正式 policy training，也不得对
模型性能作结论。

#### Gate 3 — Small Policy Rollout

只有 Gate 1 和 Gate 2 通过后，才允许 development-scale policy 控制机器人。第一阶段只做
短时间 rollout，并检查：

- 模型输出 finite 且 action range 合法；
- robot state 更新正确；
- observation-action-observation 循环可以工作；
- 不发生明显数值发散；
- action 能通过完整 adapter 链路执行；
- checkpoint reload 后行为在既定容差内保持一致。

Gate 3 不要求模型已经稳定完成整个 manipulation task。

#### Gate 4 — Development Task Evaluation

只有前三个 Gate 通过后，才进行完整 development-scale task evaluation。至少记录：

- task success；
- rollout length；
- invalid action rate；
- joint-limit violation；
- collision；
- action smoothness；
- policy inference latency；
- simulation step latency。

训练 loss 不能代替机器人任务指标。

#### Open-loop 与 Closed-loop

M2 至少必须建立基础 observation-action-observation 循环。可以先测试短 action chunk 的
open-loop execution，但正式 development evaluation 应进入 closed-loop：

```text
observe
   |
   v
policy
   |
   v
action / action chunk
   |
   v
execute
   |
   v
observe again
```

不得仅通过离线 action prediction loss 宣称机器人 policy 已完成验证。

#### Simulation Adapter

Rosetta 核心 policy 不得直接依赖 MuJoCo API，必须通过明确的 simulation / embodiment
adapter 边界连接。概念接口为：

```text
reset() -> RosettaObservation

step(RosettaAction)
    -> next_observation
    -> reward or task metrics
    -> terminated
    -> info
```

具体 simulator 的 actuator、joint index、control mode、reference frame 和物理参数必须封装在
adapter 内。未来接入其他机器人或 simulator 时，不应要求修改通用 VLA policy。

#### M2 完成条件

M2 至少满足以下条件后才能标记完成：

1. development-scale 真实 VLM backbone 已接通；
2. 数据准备流程完整可复现；
3. action space 已记录完整物理语义；
4. Scripted Action Smoke Test 通过；
5. Dataset Action Replay 通过；
6. development-scale 模型可完成正式训练；
7. checkpoint 保存、恢复和独立重新加载通过；
8. 模型输出可实际驱动 MuJoCo 中的机器人；
9. observation-action-observation 基础循环通过；
10. development task evaluation 可运行并生成规定指标；
11. 训练配置、代码 commit、dataset revision 和模型 revision 可追踪；
12. development checkpoint 可导出并具备完整 Model Card / metadata。

M2 的核心判断标准是：**Rosetta 不仅能够预测 action，还能够证明这些 action 在明确的物理
语义下实际控制仿真机器人。**

### 冻结骨干与离线特征提取

本节只适用于 `adaptation = frozen` 的实验。只要 VLM 保持冻结，就应优先把它当作离线特征
生成器，使昂贵的 backbone 计算与便宜的 policy training 解耦。LoRA 或 Full Fine-tuning
会改变 backbone representation，必须使用相应的 Online Backbone 训练路径，不得复用旧的
冻结特征来声称完成了 backbone adaptation。

- 使用 `torch.inference_mode()` 或等价 inference-only 模式，不为冻结骨干构建梯度图。
- 内存压力不明确时默认 `batch_size = 1`。
- 按 sample / frame 增量处理，不把完整 episode 一次性加载进内存。
- 已完成的 feature 及时写盘，并在样本之间释放大型输入和输出引用。
- 第一版 baseline 优先使用满足实验目的的 pooled / reduced representation，例如
  `[sample, D]`；没有明确实验需求时不得缓存 `[sample, tokens, D]` 的完整 token-level
  hidden states。
- 可以在实现明确支持时使用量化冻结推理，但精度和量化方式必须写入配置及 cache manifest，
  不得在不同 cache generation 之间静默改变。
- 训练代码可以提供 Online Backbone 和 Cached Backbone 两条路径；它们对下游暴露的
  representation contract 必须一致，核心 VLA 架构不得依赖某一种缓存实现。
- 本地不得让 9B 模型长期常驻普通训练循环。需要扩大提取规模时，先缩小 sample / episode、
  batch 和 representation，再考虑经过用户批准的外部 GPU 环境。

### 0.8B 到 9B 的复用边界

0.8B 的价值是先解决所有能够在 development scale 解决的工程和科研流程问题，而不是把
0.8B 权重“无缝转换”为 9B。

- 可以直接复用：Dataset Adapter、Action Space、Normalization、固定 split、instruction
  构造规则、训练与评估逻辑、日志、artifact schema，以及 State Encoder、Fusion 和 Action
  Head 的架构合同。
- 必须为 9B 重新生成：backbone hidden states、Feature Cache 和通常与 hidden size 直接相连的
  Backbone Projector。即使 0.8B 与 9B 输出 tensor shape 相同，也不得认为 representation
  语义兼容。
- State Encoder 权重可以作为 9B 实验的 warm start 候选；Fusion 和 Action Head 权重只有在
  contract 一致且实验明确记录时才能尝试 warm start。所有 warm start 都应与从头初始化形成
  controlled comparison，不能默认其一定更优。
- 每个 backbone 可以拥有自己的可配置 Projector，但 Projector 输出应遵守统一、可配置的
  representation contract，使下游组件不依赖 Qwen 型号或原始 hidden size。
- 从 0.8B 扩展到 9B 时，复用的是经过验证的 pipeline、实验协议和可选择的下游初始化，
  不是跨 backbone 复用 Feature Cache 或假定模型权重可以零成本迁移。

### Feature Cache 身份与复用

Feature Cache 只有在生成模型、数据和预处理链路完全一致时才可复用。每个 cache 根目录必须
包含机器可读 manifest，至少记录：

- 数据源标识和不可变 dataset revision；
- episode、frame 或 sample 标识；
- camera 或视觉字段映射；
- 模型 family、identifier 和可获得时的不可变 model revision；
- processor / tokenizer 配置或 revision；
- 图像预处理配置；
- instruction / prompt 构造方式；
- feature 提取层；
- pooling 或 token-selection 策略；
- 输出 dtype、精度或量化模式；
- feature-cache schema version。

绝不能仅凭 tensor shape 判断两个缓存兼容。任何身份字段变化后必须生成新的
revision-scoped cache，或者明确报错并停止；不得静默复用或覆盖旧缓存。Feature Cache、
模型权重、准备后的 dataset、checkpoint 和其他大型生成物默认不得进入 Git；只提交必要的
小型配置、metadata 或 manifest。

### 实验溯源与评估

Rosetta Reality 的模型工作应形成可审计的 AI-directed research loop。每次实验至少记录：

- 实验假设、验收条件和停止条件；
- 代码版本或工作区状态；
- dataset、model、processor 和 Feature Cache 身份；
- 完整配置、随机种子、split、evaluation protocol 和资源环境；
- 哪些数据选择、清洗规则、超参数和后续实验由 AI 提出；
- 用户批准了哪些下载、昂贵计算或安全边界；
- 所有人工修改、介入和例外；
- 结果、失败分析、结论和下一轮假设。

评估至少同时考虑 train / validation loss、过拟合、预测分布和 action 合法性。M2 仿真评估
还必须按任务检查成功率、碰撞、动作平滑度、invalid action、joint-limit violation 和延迟。
除非目标只是显式 smoke test，否则一次 forward、一次 optimizer step、loss finite 或
tensor shape 正确都不构成模型有效性的充分证据。

### 0.8B 实验模型验收与发布

0.8B development model 是正式研究 artifact。进入 9B scale-up 前，至少应证明：

- train / validation split 无已知泄漏，完整训练和评估使用固定 protocol；
- checkpoint 保存、恢复和继续训练可用；
- 导出的 artifact 能在独立加载路径中恢复出同一模型合同；
- 配置、seed、代码 revision、dataset revision、processor revision、日志和指标足以复现；
- Model Card 草稿明确记录用途、限制、输入、输出、action space、训练方法和评估结果；
- 模型明确标注为 experimental / research only，且未经过真机验证时不得暗示可自主控制
  物理机器人。

发布到 Hugging Face 或其他外部平台属于外部写操作，必须获得用户对目标仓库和版本的明确
授权，并在发布前核对基础模型、训练数据和代码许可证：

- Frozen backbone 实验优先只发布 Rosetta Projector、State Encoder、Fusion、Action Head、
  配置和 Model Card，并使用 `base_model` 指向原始模型；不要重复上传未修改的基础权重。
- LoRA 实验优先发布 adapter、Rosetta action components、配置和 Model Card。
- Full Fine-tuning 只有在许可证允许、artifact 验证通过且用户明确批准时才能发布完整权重。
- 发布验收必须包含可从公开 artifact 重新加载的验证；上传成功本身不等于模型可复现。

### 资源不足时的默认决策

遇到内存或计算压力时，按以下优先级调整系统设计：

1. 使用更小的开发 checkpoint；
2. 缩小 sample 或 episode 范围；
3. 降低 batch size；
4. 使用 inference-only 执行并冻结 backbone；
5. 复用经过 manifest 验证的 Feature Cache；
6. 使用 pooled representation；
7. 明确、可追溯地使用量化；
8. 拆分 CPU 与 GPU 阶段；
9. 只把范围受控的昂贵阶段迁移到用户批准的 GPU 环境。

不得依赖大量 swap、不受控的 CPU / disk offload、反复 OOM 尝试或强行扩大 WSL 内存来
维持不适合本地资源的方案。16 GB 本地内存是系统设计约束，不是降低 revision 管理、
可复现性、验证强度或 provenance 要求的理由。
