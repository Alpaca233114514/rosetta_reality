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
- M2 必须先完成冻结骨干 baseline。先使用本地已有的 `Qwen3.5-0.8B-Base` 或其他兼容的
  小型 checkpoint 验证真实多模态链路；不得为了执行此步骤自行下载权重。
- 小模型链路必须在极小样本上验证图像预处理、instruction / prompt 处理、hidden state
  提取、tensor shape、pooling、机器人状态融合和动作预测。
- 只有小模型链路通过后，才能让 9B 骨干进行范围受限的冻结推理。M2 中 9B backbone
  必须保持冻结，只训练 State Encoder、Fusion 和 Action Head。
- 冻结 baseline 在固定 dataset split 和 evaluation protocol 上可稳定复现并完成验收前，
  不得开始 LoRA、其他参数高效适配或端到端微调。
- M3 的 LoRA 必须作为独立实验，与 M2 frozen baseline 形成直接对照。真正的 9B LoRA
  原则上只在用户批准的 GPU 环境运行；本地只允许用小型 checkpoint 做有限 smoke test。
- 不得把 Frozen Backbone + Action Head Training 描述为 LoRA、full fine-tuning 或
  9B 全参数训练。

### 每次模型工作的执行顺序

每项模型改动或实验必须按以下顺序推进，并在上一步有可检查证据后才进入下一步：

1. 定义一个明确假设、变更范围、数据和模型身份、资源预算、验收指标及停止条件。
2. 先完成不加载权重和数据的静态检查、配置检查、shape / contract 单元测试。
3. 在 WSL 中运行 `python scripts/check_env.py`、相关单元测试、dummy forward 和 CPU 或
   小模型 smoke test。具体测试范围应与风险相称；进入真实 GPU 训练前仍须执行本文件前述
   完整 GPU 前置检查顺序。
4. 使用小型真实多模态 checkpoint 和极小样本验证 Online Backbone 路径。
5. 对冻结骨干执行范围受限、可恢复的离线特征提取，生成不可变 Feature Cache。
6. 在训练前验证 Feature Cache manifest、数据完整性和 representation contract。
7. 使用合法缓存训练轻量 State Encoder、Fusion 和 Action Head；相同 backbone 和输入
   observation 不得在每个 epoch 中重复 forward。
8. 在固定 split 和 protocol 上评估，并检查用户可见的任务结果；不能只凭 training loss
   下降宣告成功。
9. 记录配置、结果、失败、人工介入和下一轮假设，再决定是否扩大数据、模型或计算规模。

标准 M2 主路径为：

```text
小模型真实 Backbone Smoke Test
            |
            v
受限 Frozen Backbone Feature Extraction
            |
            v
Feature Cache Validation
            |
            v
轻量 State / Fusion / Action Training
            |
            v
Evaluation
```

任何一步失败时，应修复该层并重新验证，不得通过扩大模型、数据、内存、swap 或计算资源
来掩盖接口、数据或实验设计问题。

### 冻结骨干与离线特征提取

只要大 VLM 保持冻结，就应优先把它当作离线特征生成器，使昂贵的 backbone 计算与便宜的
policy training 解耦。

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

评估至少同时考虑 train / validation loss、过拟合、预测分布和 action 合法性。进入仿真后，
还应按任务检查成功率、碰撞、动作平滑度、invalid action 和延迟。除非目标只是显式 smoke
test，否则一次 forward、一次 optimizer step、loss finite 或 tensor shape 正确都不构成
模型有效性的充分证据。

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
