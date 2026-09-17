# G4 Seed 3 与可解释性诊断

本入口围绕固定 canonical step-5000 排查模型行为和训练代码，不改变正式
G4 seeds 1000–1004、阈值或历史证据。Seed 3 是独立环境 seed=3、噪声 seed=3
的单回合诊断，最多 500 步、50 Hz、每次只执行预测 chunk 首动作。
单次成功也不更新正式 G4 或 M2 状态。

## 入口和阶段

统一入口 `scripts/diagnose_smolvla_gate.py`；实现位于
`src/rosetta_reality/eval/gate_diagnostic_*.py`。继承原 rollout 与 trace，
历史 `diagnose_canonical_rollout.py` 的五种子限制保留。

| 命令 | 参数 | 行为 |
| --- | --- | --- |
| `validate-plan` | `--plan`，可选 `--schema-only` | 默认核对完整文件身份；schema-only 仅检查声明，不授予执行资格 |
| `collect` | `--plan` | 一次 Seed 3 rollout，捕获原始模型调用，零额外 forward |
| `replay` | `--plan --input` | 新进程逐步精确重放全部保存输入/噪声，不调用仿真 |
| `probe` | `--plan --input` | 最多 12 时点、60 次 forward，干预动作不执行 |
| `audit-training` | `--plan --output` | 核查有模型/源码绑定的历史 JSON 操作数，不加载权重/数据 |
| `analyze` | `--input --output`，可选 `--probe --training` | JSON、Markdown 和时间线 CSV |
| `verify` | `--directory`；重放/探针另加 `--source` | SHA/结构验证和独立 NumPy 数值复算，不加载模型 |

模型阶段写入计划 `output` 下的 `collect`、`replay`、`probe` 子目录。
每个目录只创建一次；失败保留，禁止覆盖或自动重试。重放/探针必须与采集的
模型、processor、合同和源码一致，并使用独立 CLI 进程。`collect`、`replay`、
`probe` 的真实 CUDA 运行都需要新的有效授权窗口。

```bash
# 下列命令在固定 Linux Docker 内执行；Windows 通过 wsl.exe bash 启动 Docker。
python scripts/diagnose_smolvla_gate.py validate-plan \
  --plan configs/vla/gate_seed3_diagnostic_001.json --schema-only

python scripts/diagnose_smolvla_gate.py audit-training \
  --plan configs/vla/gate_seed3_diagnostic_001.json \
  --output runs/NEW_DIAGNOSTIC/training-audit

python scripts/diagnose_smolvla_gate.py verify --directory runs/NEW_DIAGNOSTIC/collect
python scripts/diagnose_smolvla_gate.py verify \
  --directory runs/NEW_DIAGNOSTIC/probe --source runs/NEW_DIAGNOSTIC/collect
python scripts/diagnose_smolvla_gate.py analyze \
  --input runs/NEW_DIAGNOSTIC/collect --probe runs/NEW_DIAGNOSTIC/probe \
  --training runs/NEW_DIAGNOSTIC/training-audit --output runs/NEW_DIAGNOSTIC/analysis
```

`validate-plan` 的成功仅证明指定级别的校验完成，`execution_ready` 保持 false；
真正运行还要通过 stage 授权、环境、watchdog、存储和资源门禁。
失败退出码为非零。保存文件可验证不等于任务成功。

## 真实运行注册

`configs/vla/gate_seed3_diagnostic_001.json` 是明确不可执行的 draft；
含 canonical 文件清单及两个历史报告索引，缺少的执行路径、源码清单、授权
和运行时身份保留 null/空值，不填入虚构的通过证据。

后续运行创建新名字和新注册，不修改历史计划。补齐：

- `status=registered`、唯一 `run_id`、`output=runs/...`。
- `sources`：`gate_diagnostic_protocol.REQUIRED_SOURCES` 及实际依赖的 SHA。
- `checkpoint.path/files`、`training_plan`、`artifact_config`、`action_contract`。
  文件引用采用 `{path, sha256}`；全部使用仓库相对 POSIX 路径，拒绝越界。
  step-5000 权重及全部 processor 清单固定于 canonical attempt 004。
- `artifact_manifest`、`gate3_report`：匹配同一 artifact 的已验证独立 reload
  和已通过 G3。原 native loader 继续执行自身的数据、processor 和前置门禁。
- `execution.authorized=true`、`stages`（只允许本次获准阶段）、`deadline`
  （Unix 秒，未来不超过一小时）、`watchdog={pid,ticks}`。
- `execution.runtime=autodl_cuda_container_instance`、`nested_docker_used=false`、
  `gpu_uuid`、`package_versions`（torch、lerobot、gym-aloha、mujoco 精确版本）。
  环境必须离线并声明 CUDA。CLI 不开机、不启动 watchdog、不关机或接管别的任务；
  继续使用既有受保护 worker 的启动、截止、回传和关机生命周期。

每阶段默认证据上限 4 GiB（可显式收紧到最低 64 MiB），启动前要求完整预算
加 256 MiB 磁盘余量。RSS 8 GiB、CUDA allocated 8 GiB/reserved 10 GiB，
每次预测前后检查截止和资源；超限保留部分结果并停止。图片和完整 batch
按步流式写盘，压缩只影响存储，不改变张量。每次写入预留 trace/最终清单空间。

## 证据和解释边界

每个包含 `identity.json`、`result.json`、`manifest.json`；采集另含原始
`trace/` 和 `predictions/0000/...`。每次预测的 `tree.json + arrays.npz`
保存精确 dtype/shape 和 uint8 字节，不使用 pickle，支持 BF16。

保存内容：原始相机/state/指令、实际 preprocessor 输出 batch、实际完整 noise、
模型归一化输出、内部动作、decoder 输出和 projection 输出；trace 保存动作执行、
实际 state/位姿、contact、reward、终止、限位和内部夹爪支持域。
模型噪声宽度取自 policy 配置，不按物理动作维度截断。

采集包装器不改输入/返回值、不增加 forward/随机采样，不重跑 processor。
成功时 RNG 按原流程推进；失败和离线重放恢复入口 RNG；方法绑定在 finally 恢复。
重放通过原在线适配器重新处理原始 observation，在模型调用边界注入保存的 noise。
原适配器生成的临时噪声不用于模型，重放结束恢复 RNG，不影响采集轨迹。

重放按 observation → instruction → batch → noise → normalized → internal →
decoded → projected 定位首个不一致边界。两种控制（原样和自拷贝）均通过后，
分别替换前一步 image/state/noise；step 0 使用下一步。无不同 donor 明确跳过。
probe 选择 step 0、首次物体接触/接触丢失/reward 变化及前后一步，排序截取后
用均匀时间点补足到最多 12 个。接触丢失指此前观察到的 robot-object geom pair
消失，不自动等同于真实抓取滑落。

四个动作组分别报告首动作/完整 chunk 的平均与最大绝对变化及物理单位。
输入替换可能离开真实数据分布，只说明局部依赖。无专家轨迹时不计算专家误差、
不把 step 0 定为首次错误。接触不等于稳定抓取，命令位姿不等于实际到达位姿。
缺失物理字段为 null/CSV 空白；不补零。不完整运行的 task_success 为 null。
耗尽步数且证据完整、task_success=false 才是已测单回合负结果。

## 训练代码证据接口

`training_evidence` 是历史 JSON 索引。每项包含 `{path,sha256,binding,fact_pointers}`。
`binding` 用 JSON Pointer 指向报告自身的 `checkpoint_pointer`、
`training_plan_pointer` 和 `sources_pointer`。源码清单必须非空且逐文件验证；
旧源码可通过 `archived_sources` 的原路径→封存路径映射验证，不能改历史 SHA。
每项检查默认绑定检查表中的代码入口；报告使用其他已登记实现时，
`check_sources` 可显式将检查名映射到该历史 `sources` 清单中的非空源码路径列表。
默认两个摘要报告缺少完整可消费绑定，审计会诚实标记证据不足。

`fact_pointers` 把以下检查名映射到报告中保存的操作数对象，不消费笼统的 passed：

| 检查名 | 必需操作数 |
| --- | --- |
| sample_consumption / camera / pixels / processor / reload | expected、actual（完整身份/数组或哈希；调用方显式选择同义量） |
| time_chunk_padding | episode、frame、episode_length、chunk_length、target_indices、target_episodes、is_pad、input_timestamp、expected_timestamp |
| split | train、development、hidden、materialized episode 清单 |
| loss | weights、is_pad、action_dim、denominator（单样本有效项归约） |
| horizon | training_horizon、executed_horizon；不一致是合同风险，不证明代码错误 |
| freeze | expected_trainable、actual_trainable、changed_parameters、frozen_parameters |
| optimizer | trainable_parameters、optimizer_parameters（检测缺失和重复） |
| scheduler | optimizer_updates、scheduler_updates、expected_lr、actual_lr |
| gradients | preclip_norm、postclip_norm、clip_limit、nonfinite_count |

旧检查点之间的对比保留原始身份；step-2500 不能伪装成 step-5000 的代码通过证据。
没有记录的梯度、resume 或实际更新信息保持未验证，不反推。审计不构造 optimizer。
报告区分代码合同反例、模型行为、合同风险和证据不足，并记录源码入口和下一项
最小验证；任何分类均不自动宣称唯一根因，也不触发修复、选模型或训练。

## 本地验证

使用现有固定镜像
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`，
网络关闭、CPU 2、内存与 memory+swap 均 3 GiB、仓库只读。
运行 `scripts/check_env.py`、新增 `tests/test_gate_seed3_diagnostics.py` 与
原 trace/action/processor 相关回归。测试使用 tiny CPU 张量和 fake environment，
不加载模型权重、真实数据或 MuJoCo 仿真。

真实 CUDA 全数组重放、采集与未采集的实际轨迹等价性、Seed 3 任务结果均须
后续单独登记实测。合成测试不能代替这些验收。

2026-09-17 本地验收：**109 passed、1 skipped**（CUDA 专用像素一致性检查），
Ruff 与 `git diff --check` 通过。13 步 synthetic 采集、逐步重放、12 时点/60 次
探针、训练证据审计和报告包均通过校验。日志、JUnit 和明确标注 synthetic 的
示例报告位于 `runs/gate-seed3-framework-local-20260917-001/`；其中
`synthetic/analysis/report.md` 可用于查看报告格式。摘要与限制见
`reports/training/m2-smolvla-seed3-framework-local-2026-09-17.md`。
