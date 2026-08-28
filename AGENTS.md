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

## Shell and runtime boundary

- Windows-side repository editing, Git operations, and non-ML static checks may use PowerShell (`pwsh`).
- Except for PowerShell, all command-line operations must use Bash; do not substitute `cmd.exe`, another
  shell, or native Windows Python.
- All WSL operations must be executed inside WSL Bash. This includes creating or updating Python
  environments, installing ML or data dependencies, downloading models or datasets, and running ML code.
- When invoking WSL from PowerShell, call `wsl.exe bash` explicitly and quote Bash variables and paths so
  PowerShell expansion cannot redirect a command into the Windows-mounted repository unexpectedly.
- 正式的数据、模型、训练、评估和仿真命令必须从 WSL Bash 启动，并在 Linux Docker 容器中
  执行。WSL 主机只负责只读检查、依赖源码检出、容器编排和受控的 Hub 控制面操作；不得把
  主机 Python 环境当作训练环境。
- AutoDL 容器实例是上述规则的唯一已登记远端例外：平台实例本身作为 Linux 容器边界，禁止
  尝试嵌套 Docker。必须使用 `configs/runtime/autodl_rtx4090.yaml` 与
  `scripts/run_autodl.sh` 记录 `nested_docker_used=false`，先通过远端 doctor、immutable cache
  identity、pre-training benchmark、no-optimizer CUDA forward 和两步 CUDA optimizer smoke，
  再由单独 preregistered formal plan 授权正式训练。不得把 AutoDL shell、WSL 主机 Python 或
  未经身份核验的云端环境当作等价容器证据。
- 可以从 WSL 使用已经登录的 Hugging Face CLI 创建或检查 Space、模型仓库和数据仓库；模型
  或数据内容的准备与训练仍须在容器中完成。不得读取、打印、写入配置文件或通过命令行参数
  暴露 Hugging Face token。

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

## AutoDL 远程炼丹炉强制工作流

AutoDL 是临时 CUDA worker，本地 Codex、仓库和 WSL 是 control plane。除非用户明确改变方案，
所有 AutoDL 训练必须固定遵守以下流程：

```text
Local control plane
  edit / inspect / benchmark plan / freeze identities
        |
        | immutable Git revision or versioned workspace bundle
        v
AutoDL CUDA worker
  doctor -> benchmark -> no-optimizer forward -> two-step smoke
        |
        v
  tmux guarded train -> validate -> select -> export
        |
        | selected artifact + manifests + reports
        v
Local control plane
  independent reload -> shut down paid GPU -> local Gate 3 / Gate 4
        |
        v
  diagnose -> preregister the next single-variable run
```

### 本地控制面与远端 worker 边界

- 本地负责阅读文档、修改和审计代码、比较实验、冻结假设与变量、生成计划、核验 Git/workspace
  identity、分析 Gate 结果以及决定下一炉。不得把 Codex 配置、长期开发状态或决策过程迁移到
  AutoDL，也不得在计费 GPU 上临时探索研究方向。
- 开机前必须在本地冻结：实验假设、唯一 run name、代码身份、config/plan checksum、dataset/model/
  processor revision、Action Contract、optimizer/scheduler、batch/steps、benchmark 命令、smoke 命令、
  输出目录、验收指标、停止条件和预计时长。上述任一项未冻结，不得开启正式 GPU 训练。
- 代码传输优先使用已推送的普通功能分支 commit；用户尚未授权 commit/push 时，只能使用
  `scripts/stage_autodl_from_wsl.sh` 创建新的 versioned、content-addressed workspace。禁止手工复制
  若干 `.py` 文件、覆盖既有远端 workspace、在 run 启动后编辑远端代码，或让同一 run 跨两个
  workspace identity。
- 数据、模型、VLM dependency 与 processor cache 必须位于远端 durable data root，保留原有
  revision-scoped 目录和 manifest；传输不得使用 `--delete`，不得覆盖不同身份缓存，不得因远端
  缺少缓存而自行下载。doctor 必须在不加载模型权重和数据行的情况下先验证所有 manifest。
- AutoDL GPU 只承担经过授权的 CUDA doctor/benchmark/smoke、昂贵训练、必要 validation、selection、
  export 和 independent reload。固定等待、长时间 Gate 3/4、实验分析、代码调试、文档整理与下一炉
  设计默认回到本地执行，不得让计费 GPU 空转等待。

### 启动、守护与固定五分钟阻塞

- 任何长于短 smoke 的远端进程必须在 `tmux` 或等价守护会话中运行，stdout/stderr 同时写入
  durable run log；SSH 连接只负责控制，绝不能承担训练进程生命周期。SSH 或本地 Codex 断开后，
  训练必须继续，重新连接后只能通过已记录的 session/run identity 恢复观察。
- 启动后先检查进程、GPU、首批 step、finite loss/gradient、显存、Trackio 和 checkpoint 目录；
  一旦确认进入稳定训练，负责监控训练的 agent 必须使用**固定五分钟阻塞**：每次等待只能执行
  Bash `sleep 300`，不得改成其他秒数，不得使用短轮询、连续 `tail -f`、高频 GPU 查询、忙等或
  无阻塞反复检查。`sleep 300` 结束后只做一次有界状态采样，再继续下一次 `sleep 300`。
- 固定五分钟阻塞不改变已登记的 quarter/checkpoint wake gate：只有在启动健康检查、失败信号、
  预登记 checkpoint/25%/50%/75%/100% 边界或训练完成时才做完整审计；其余五分钟唤醒只允许读取
  最小状态，不得据此修改超参数、重启进程或开启另一炉。
- `sleep 300` 阻塞的是监控 agent/控制 shell，不得注入训练 dataloader、optimizer、scheduler、
  simulator step、checkpoint writer 或训练进程本身；不得因阻塞丢失日志、心跳、异常退出状态或
  durable metrics。训练进程退出后必须立即停止新的 sleep 周期并进入结果核验。

### 训练完成、回传与关机

- 每个远端 run 必须形成黑匣子，至少包含 resolved config/plan、Git revision 或 workspace tree hash、
  dataset/model/processor revisions、Action Contract、完整 optimizer/scheduler contract、环境与 GPU
  身份、train log、Trackio identity、metrics、checkpoint manifest、selection report、export manifest、
  人工介入和退出状态。不得只留下 `model.safetensors` 或口头结果。
- 完成顺序固定为：训练退出并保存状态 -> validation-only selection -> export -> remote independent
  reload -> 回传 selected deploy artifact、manifest、metrics 与必要报告 -> 本地 checksum/reload 核验
  -> 在用户已授权的运行边界内关闭计费 GPU -> 本地 Gate 3 -> 本地 Gate 4 -> 结果分析。任一步失败
  都不得假装完成、删除远端状态或直接启动下一炉。
- 完整 optimizer/scheduler checkpoint 默认保留在远端用于显式 resume；selected deploy artifact 与
  provenance 必须拉回本地。若实例可能释放、迁移或不再保留，必须在关机/释放前把需要复现或恢复
  的完整 checkpoint 备份到用户批准的可靠位置。AutoDL 本地盘不构成唯一可靠副本。
- Gate 3/4 默认在本地通过独立 reload 的 deploy artifact 运行。固定 Gate 4 等待期间不得保持
  AutoDL GPU 开机；不得用远端更快的离线 loss、一次 positive reward 或一次偶然 success 替代本地
  registered Gate 结果。
- 下一炉只能在本地分析当前 run 后，重新提出单一受控假设并生成新的 identity/plan；禁止在远端
  原地改参数续跑、复用不匹配 optimizer state、无计划批量开炉，或把 AutoDL 变成常驻开发机。

## Architecture

### Current M2 navigation (mandatory)

- Any task that touches SmolVLA data, `configs/vla/`, `src/rosetta_reality/vla/`,
  the SmolVLA trainer/loss/optimizer/scheduler, checkpoint/resume, export,
  validation, Gate 3 or Gate 4 must first read
  `docs/m2-smolvla-architecture.md` completely. It is the stable component and
  control-flow map; do not reconstruct the architecture from chat history.
- After the architecture map, read
  `reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md`
  and its JSON companion for the findings framework and repair order, then the
  newest completed campaign audit
  `reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md` and its JSON
  companion for the current evidence state. The earlier `docs/er-vla-pipeline.md`
  and action-repair handoff remain context and provenance, not current
  completed-result authority.
- Current boundary: Gate 4 has failed `0/5` under one identical protocol for
  five identities — Faust, Aster, Way, Zen-uniform (`411`) and Zen-firstaction
  (`422`); every other stage (training, selection, export/reload, Gate 3)
  passed for all of them. Therefore M2 is not complete. No agent may infer
  acceptance from offline MAE or start another full furnace without a new
  single-axis registered plan and the required authorization/gates. The
  registered follow-up diagnostic is the Zen first-deviation trace
  (`reports/training/m2-smolvla-zen-first-deviation-preregistration-2026-08-28.md`),
  pending deploy-artifact transfer; the AutoDL instance is shut down and must
  not be released.
- If architecture prose conflicts with a hash-bound config, Action Contract,
  executable assertion or immutable evidence, stop and reconcile the mismatch.
  Never silently choose one or edit historical evidence to match prose.
- When component ownership, entry points, the current evidence source, processor
  boundary, trainer/optimizer architecture, closed-loop flow or Gate 4 status
  changes, update `docs/m2-smolvla-architecture.md` in the same change. If its
  stable path changes, update this section, `README.md` and
  `docs/architecture.md` together.

- Rosetta Reality 是一个 **Embodied Reasoning + Vision-Language-Action monorepo**。ER、VLA 与
  integration 在逻辑上分离，但共享一次 Git revision、数据合同、仿真适配器和评估协议。
- `Qwen3.5` 只属于 ER / System-2 路线，负责低频场景理解、任务分解、进度判断和失败恢复；
  不得再把 Qwen frozen feature + MLP action head 当作当前 VLA 主模型。
- `lerobot/smolvla_base` 450M 是当前 VLA / System-1 development model，负责从图像、语言、
  robot state 产生连续 action chunk。SmolVLA 与 Qwen 的 checkpoint、optimizer、Feature Cache、
  模型输入和发布 artifact 必须分开管理。
- 旧的 `configs/experiments/m2_qwen08b_*`、对应报告、checkpoint、Feature Cache 和 run 是不可变
  历史 VLA 实验及负结果证据。除非另有明确迁移方案，只允许只读复用其数据身份、split、
  Action Contract、Gate 1/2、仿真适配器、指标和诊断结论，不得复用其模型权重或特征作为
  SmolVLA 初始化。
- ER 到 VLA 的唯一正式边界是版本化 structured `ActionPlan`。第一版至少包含 `subtask`、
  `object`、`target`、`motion_hint`、`constraints`、`success_condition` 和 `replan_condition`；
  不得把一句无结构自然语言当作完整 ER/VLA 接口。
- Keep both the ER backbone and VLA policy replaceable.
- Keep the dataset layer robot- and embodiment-agnostic.
- Do not leak Qwen-specific behavior into VLA or integration components, and do not leak SmolVLA-specific
  behavior into ER or generic simulation/data contracts.
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
- ER 模型、VLA 模型、backbone adaptation、Action Expert、dataset 和 ER/VLA interface 是相互
  独立的实验轴。不得用其中一条线的成功替代另一条线的验收。
- **M2 — SmolVLA Development VLA**：使用 revision-pinned `lerobot/smolvla_base` 450M 完成
  第一代 Rosetta VLA 的数据、训练、验证、checkpoint / resume、evaluation、artifact export、
  独立 reload 和 MuJoCo 闭环。M2 完成时必须存在能在明确 action 语义下驱动目标仿真机器人的
  SmolVLA checkpoint，而不只是 finite loss 或 action tensor。
- M2 首选复用 revision-pinned `lerobot/aloha_sim_insertion_human` 50 episodes、既定 split、
  Action Contract 与已经通过的 Gate 1/2 证据。复用前仍须在当前容器和当前配置下只读核验
  manifest、checksum、camera/state/action/task 字段、fps、chunk 语义和 split；不得仅凭 repo_id
  或 shape 宣称兼容。
- 每种新的 SmolVLA 训练能力必须从 `batch_size = 1`、极少 step 和固定小样本开始，依次验证
  processor、camera mapping、instruction、state/action padding、normalization、forward、backward、
  optimizer、checkpoint、resume、Trackio 和小数据 overfit，再依据实测资源决定是否扩大。
- **M3 — ER/VLA Integration**：只有 SmolVLA M2 闭环通过，且 Qwen ER 在独立 ER 数据和指标上
  通过验收后，才能通过版本化 `ActionPlan` 接入。必须分别报告 ER plan quality、VLA execution
  quality 和 end-to-end success，不能只报告最终成功率。
- Qwen 0.8B 到 9B 属于 ER scale-up，不是 VLA 里程碑。9B 的 LoRA 或 Full Fine-tuning 只能在
  用户批准且经过资源预算验证的 GPU 环境运行；本地 16 GB 机器不承担 9B 正式训练。
- 不得把 SmolVLA 的 `freeze_vision_encoder` / `train_expert_only` 描述为 LoRA 或 full fine-tuning，
  也不得把旧 Qwen action-head checkpoint 描述为 ER checkpoint。

### 每次模型工作的执行顺序

每项模型改动或实验必须按以下顺序推进，并在上一步有可检查证据后才进入下一步：

1. 定义明确假设、工作线（ER / VLA / integration）、变更范围、不可变数据/模型/上游代码身份、
   adaptation、资源预算、验收指标和停止条件；不得混淆实验变量。
2. 先完成不加载权重和数据的静态检查、配置检查、schema / shape / contract 单元测试。
3. 从 WSL Bash 构建或检查 digest/revision-pinned Docker image，在容器中运行环境检查、相关单测、
   dummy forward 和 CPU smoke；不得用 Windows Python 或主机 WSL Python 代替容器证据。
4. 正式 VLA 训练前定义完整 Action Contract 和 Simulation Adapter，并依次通过 M2 Gate 1
   Scripted Action Smoke Test 与 Gate 2 Dataset Action Replay。
5. 创建并验证 Trackio Hugging Face Space；正式 run 必须预先固定 project、Space、run name、
   config identity 和本地持久化目录。动态 Space 不可用时，允许先写本地 durable store，再在
   checkpoint 边界把经过脱敏检查的指标同步到公开 static Space；不得丢弃 metrics 或假装实时。
6. 使用 SmolVLA 450M、`batch_size = 1`、极少 step 和极小真实样本验证端到端 forward / backward、
   峰值内存、step latency、梯度、checkpoint 和 Trackio；非 finite、OOM 或不受支持的 accelerator
   必须立即停止，不得靠反复 OOM 或 swap 掩盖。
7. 使用固定小样本集验证 SmolVLA 能够 overfit，并排查 split 泄漏、processor、normalization、
   checkpoint 保存/恢复、日志和 evaluation。
8. 资源实测通过后，才完成正式 training / validation、显式 resume、evaluation、artifact export、
   独立 reload 和可复现性验证。
9. 依次通过 M2 Gate 3 Small Policy Rollout 与 Gate 4 Development Task Evaluation，证明
   checkpoint 经完整 adapter 链路形成 observation-action-observation 控制循环。
10. Qwen ER 必须使用独立 ER 配置、监督信号、checkpoint 和 benchmark；旧 VLA action loss 不能
    作为 ER 训练或 ER 能力证据。
11. 只有 ER 与 VLA 各自通过后，才接入 `ActionPlan`，依次做 schema round-trip、grounding、单步
    执行、replan/failure recovery 和端到端评估。
12. 记录配置、结果、失败、Trackio URL、人工介入和下一轮假设，再决定是否扩大数据、模型或计算。

主路径为：

```text
Static / Schema / Contract Tests
        |
        v
M2 Gate 1 Scripted Action + Gate 2 Dataset Replay
        |
        v
Trackio Space + SmolVLA 450M Tiny Smoke
        |
        v
SmolVLA Small-data Overfit
        |
        v
SmolVLA Train / Validate / Resume / Evaluate / Export
        |
        v
M2 Gate 3 Small Rollout + Gate 4 Task Evaluation
        |
        v
Qwen ER Independent Training + ER Evaluation
        |
        v
ActionPlan Integration Tests
        |
        v
ER + VLA End-to-end Evaluation
```

任何一步失败时，应修复该层并重新验证，不得通过扩大模型、数据、内存、swap 或计算资源
来掩盖接口、数据或实验设计问题。

### M2 — Development VLA 与仿真控制验证

M2 的目标不是仅让模型成功输出动作张量，而是建立第一条从真实多模态 observation 到仿真
机器人实际运动的完整控制链路。M2 固定使用 revision-pinned `lerobot/smolvla_base` 450M，
完成端到端训练、评估、checkpoint、恢复、导出及仿真控制验证。Qwen 不参与 M2 VLA checkpoint。

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
- 旧 Qwen frozen Feature Cache 只可用于复现历史 VLA 诊断或新的 Qwen ER 假设，绝不能作为
  SmolVLA 输入、初始化或 SmolVLA 训练完成证据。SmolVLA 主路径使用其原生 online processor
  和训练合同；`freeze_vision_encoder` 不等于离线 Qwen Feature Cache。
- 本地不得让 Qwen 9B 模型长期常驻普通训练循环。需要扩大提取规模时，先缩小 sample / episode、
  batch 和 representation，再考虑经过用户批准的外部 GPU 环境。

### Qwen ER 0.8B 到 9B 的复用边界

本节只适用于 Qwen ER，不适用于 SmolVLA。0.8B 的价值是先解决 ER supervision、prompt、
structured plan、评估和 integration contract，而不是把 0.8B 权重“无缝转换”为 9B。

- 可以直接复用：ER dataset adapter、固定 split、instruction / reasoning supervision 规则、
  ActionPlan schema、ER 训练与评估逻辑、日志和 artifact schema。
- 必须为 9B 重新生成：backbone hidden states、Feature Cache 和通常与 hidden size 直接相连的
  Backbone Projector。即使 0.8B 与 9B 输出 tensor shape 相同，也不得认为 representation
  语义兼容。
- ER Projector 或 adapter 权重只能在合同一致且实验明确记录时作为 warm start 候选；所有
  warm start 必须与从头初始化形成 controlled comparison。
- 从 0.8B 扩展到 9B 时，复用的是经过验证的 pipeline、实验协议和可选择的下游初始化，
  不是跨 backbone 复用 Feature Cache，也不是把任一 Qwen 权重迁移给 SmolVLA。

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
- Trackio project、Space、run name、run URL、本地存储身份和 sync 状态；token 不属于 provenance，
  不得写入日志或 manifest。
- 公开 Trackio Space 只允许同步指标、非敏感超参数、不可变 model/dataset/code revision、资源统计
  和 run 状态。不得同步 token、环境变量、绝对主机路径、原始样本、私有对话、完整控制台日志、
  checkpoint、模型权重或未经过明确审查的媒体/artifact。

评估至少同时考虑 train / validation loss、过拟合、预测分布和 action 合法性。M2 仿真评估
还必须按任务检查成功率、碰撞、动作平滑度、invalid action、joint-limit violation 和延迟。
除非目标只是显式 smoke test，否则一次 forward、一次 optimizer step、loss finite 或
tensor shape 正确都不构成模型有效性的充分证据。

### SmolVLA 与 Qwen ER 模型验收和发布

SmolVLA development model 和 Qwen ER model 是两类独立研究 artifact。任何一类对外发布前，
至少应证明：

- train / validation split 无已知泄漏，完整训练和评估使用固定 protocol；
- checkpoint 保存、恢复和继续训练可用；
- 导出的 artifact 能在独立加载路径中恢复出同一模型合同；
- 配置、seed、代码 revision、dataset revision、processor revision、日志和指标足以复现；
- Model Card 草稿明确记录用途、限制、输入、输出、action space、训练方法和评估结果；
- 模型明确标注为 experimental / research only，且未经过真机验证时不得暗示可自主控制
  物理机器人。

发布到 Hugging Face 或其他外部平台属于外部写操作，必须获得用户对目标仓库和版本的明确
授权，并在发布前核对基础模型、训练数据和代码许可证：

- SmolVLA 优先发布实际修改的 policy artifact、processor、训练配置、Action Contract 和 Model
  Card，并用 `base_model` 指向 immutable `lerobot/smolvla_base` revision；不得重复上传未修改
  的基础权重。
- Qwen ER Frozen 实验优先只发布 ER Projector / head、配置和 Model Card；LoRA 实验优先发布
  adapter。两类 artifact 的 repo_id、tag 和 Model Card 不得混用。
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
