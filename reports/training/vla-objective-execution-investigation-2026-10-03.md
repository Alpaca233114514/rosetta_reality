# Qwen / SmolVLA 目标与实际执行调查（2026-10-03）

结论：**SmolVLA 已重复出现“离线目标改善、任务闭环失败”的现象，但当前证据不能证明它与旧 Qwen 是同一个根因，也不能证明继续训练或放弃 VLA 是正确决策。** 本轮优先定位实际失败边界。003 的 2445 接续已经执行首个物理步，再因物理安全事件停止；源 teacher 首次纠正明确进入夹爪闭合，而三臂学生首槽仍较开放。下一步应先做同状态、同噪声、有效 processor 可观测的最小对照，不能直接重训。

本次仅进行源码阅读、保存 JSON 的数学复算、SHA 核验和 PowerShell 静态证据回归；未加载模型、原始数据集、NPZ 样本或 safetensors 张量，未启动推理、训练、仿真、SSH、付费计算、下载、发布、commit、push 或 PR。哈希 processor stats 文件不等于解码或运行 processor。没有修改核心实现、安全阈值或历史报告。

## 1. 当前工作区与证据身份

- 工作目录是当前 Rosetta 仓库；实际分支 `codex/paired-003-evidence-audit-20260930`，HEAD `66a5bbcdbdcdba554aac6d99171d8cbb501eb208`。连接短暂中断后再次核对，均未变化；没有切换、重置、拉取或覆盖本地以追赶远端。
- 调查开始已有 staged 修改：`docs/aloha-cohort-qualification.md`、`scripts/audit_targeted_aloha.py`、`src/rosetta_reality/diagnostics/cohort_qualification.py`、`tests/test_execution_qualification_stdlib.py`；`docs/m2-smolvla-architecture.md` 同时 staged/unstaged；`src/rosetta_reality/vla/training/features.py` 已有 unstaged 两行，注册 `PairedMixtureFeature`。大量 `.work/`、配置、工具、测试和报告原本 untracked，全部保留。HEAD 不能单独代表这批实验源码。
- 003 测量输入：`.work/paired-ABC-measurement-20260928-003-received-20260930-001/files`，其 `remote.sha256` SHA 为 `efcc8cb4ae0d5dd509b57be8a87933afbb1dfc51d12b8fa84673aa6c85b7bfc3`。本次重新核验 **151/151 文件与精确成员集合**、**14/14 evaluation 登记源码**、**15/15 三臂 checkpoint processor/config 文件**。14 个源码的覆盖范围是评价入口，不能外推为全部 trainer/upstream 已在本次独立重审。
- 源 bank manifest / audit 与三臂 plan 的 SHA 一致；本次另核验 2445 的 **8 个源文件**（R trace/commands/labels/summary 与四个 branch 文件）。源码和文件 SHA 明细见机器读出，不靠分支名、旧记忆或截图认定身份。
- 本机项目相关 memory 索引只用于定位材料；项目内 `.agents/skills`、`.codex` 不存在。本任务未找到需要调用的项目专用调查技能，不套用安全扫描技能替代实验诊断。项目范围 `rg --files -g '*diagnosis_2026-10-02.md'` 无匹配；未广搜私人目录。10/2 云端结论作为用户提供的线索，不当新实验结果。

最终机器读出：[paired-boundary-independent-check-2026-10-03-002.json](paired-boundary-independent-check-2026-10-03-002.json)。无 processor/parity 扩展的首版 `paired-boundary-independent-check-2026-10-03.json` 也保留；以 002 为本报告依据，两版不是两次物理实验。

## 2. 相较 9/30 新增的定位

### 2.1 首步停止的具体边界

`scripts/evaluate_paired_policy_v2.py:165` 恢复 branch，166 取保存 observation，167 重置该场景 noise generator；175 在模型调用前检查物理限位/碰撞，178 才预测。183 对 raw 首槽裁剪，184 调用 `env.step`，186–194 取安全/几何并写入第一行，197–198 才停止。

2445 A/B/C 原始 JSONL 均有且只有 **第 1 行**，`step=1`、`raw_clips=0`。故在已核对的源码与记录契约下，三臂通过了预测前安全检查，完成一个物理步，随后 A 记录限位数量 2，B/C 记录右臂两指碰桌。可排除“这三条没有执行任何动作、仅 reset/restore 前置拒绝”的解释。不能据此证明恢复初态、有效输入或触发关节已完整记录。

原行：上述测量根下 `measurement-{A,B,C}/recovery-bank/bank-003b-scene-2445.jsonl:1`。原 safety 实现 `src/rosetta_reality/sim/gym_aloha.py:108` 比较全部 limited joint 的实际 qpos 与模型 range、容差 `1e-5`；不是只检查 14 维命令。日志只保存数量，没有唯一违规 joint/value/range，**A 的具体触发关节仍未知**。B/C 的 contact pair 名称可定位右夹爪碰桌，但不能由它反推物体姿态、接触力或完整因果顺序。

### 2.2 将 2445 首次纠正对齐到源动作，而不是摘要

源 `.work/paired-bank-v3-20260927-001-independent-received/extracted/collection/bank-003b-scene-2445/R/trace.jsonl:178` 是 0 起算 step177 的人工偏离；**第179行、step178** 才是恢复初态对应的下一条 teacher 命令。本次逐行核对全293行安全，最终 step292 成功。源首次纠正前两侧 `grasp_contact=false`；这是“开始建立抓取”的局部对照，不是“已经抓住而保持失败”的记录。

| 2445 首槽 | teacher | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| left_gripper，0闭/1开 | 0.0900000 | 0.6054487 | 0.5735606 | 0.5901074 |
| right_gripper，0闭/1开 | 0.0900000 | 0.4011271 | 0.5891065 | 0.6010987 |

源 right_forearm_roll 状态为 `2.793635845184326`，teacher 下一命令 `2.7858126163482666`，A 为 `0.1134498193860054`：A 相对源状态跳变 `-2.6801860257983208 rad`，teacher 为约 `-0.00782323 rad`。B/C 的较大臂命令跳变仍在 left_wrist_rotate（约 +0.421/+0.423 rad）。完整14维逐值差见002 JSON。

这是比9/30“值得检查的命令跳变”更具体的抓取时机和目标动作线索。**不将 teacher 下一动作当唯一正确答案**：teacher 有额外几何/历史信息，且003当次 pre-action 初态未落盘；源状态只是比较参照，不能冒充学生实测初态。也没有证明夹爪差异或A的大跳变就是触发安全事件的唯一原因。

### 2.3 A/A 的已验证部分与未验证部分

源 `branch-restore.json.gz` 保存 `passed=true`、`engineering_steps=2`，两个 `exact_replays` hash 精确相同，本次读取并核对。`scripts/collect_paired_recovery_v3.py:162` 的 parity 恢复完整 integration/Python 状态，用 **当前 robot_state 作为 hold command** 两次推进，比较 physics、images hash 与 Python receipt，再恢复。这证明源采集时一个特定 hold command 的局部重复性；没有模型、Gaussian noise 或学生推理。

因此不能写“从未做任何A/A”，也不能写“已验证同状态 teacher/学生闭环A/A”。003缺少同模型同噪声的重复接续、旧/加observer入口 parity，以及首动作前完整恢复边界。`scripts/restore_paired_branch_v2.py:114` 确有 integration 回读、121 有 Python receipt 回读，但124–130直接返回保存的 observation；**没有重新取实时图像/状态并与保存 observation 对照**。这是一项待证的同步边界，不是已确认的 stale-observation bug。

### 2.4 teacher 正控制为什么仍不是学生能力上界

`src/rosetta_reality/sim/scripted_teacher_observer.py:155` 构造 teacher 观测，167–190包含 EEF site、peg/socket 真值位姿、reward 和 contact events。`src/rosetta_reality/sim/geometric_teacher.py:204`、272使用阶段/停滞历史；源采集 `scripts/collect_paired_recovery_v3.py:199` 重建 teacher history，359明确检验历史决策。

学生 `scripts/smolvla_autodl_vfunfreeze_sim_gate.py:328` 只接 top image、14维逻辑 state 和固定 instruction，362每次预测 reset policy。teacher 成功可以证明该源仿真状态有一条成功路径，但不证明这些额外信息能从单帧学生输入唯一恢复，也不证明“teacher监督相同输入就是可学习的单值映射”。恢复 teacher 时只 restore 环境并新建 fresh teacher 会改变其内部阶段/历史，不能用这样的失败推翻旧正控制。

### 2.5 base 的“未测”与“测过”指不同协议

本地 `reports/training/base-five-scene-independent-result-2026-09-27.md:5` 及其保存结果已记录 base 独立测量：五场景2/2/25/2/2步，共33步，0/5观察成功，全部原安全规则停止。它标明 `formal_gate_result=null`、`original_gate4_protocol_equivalent=false`。`reports/training/m2-smolvla-prepost-analysis-and-next-plan-2026-09-24.md:29` 的 **base 正式Gate4未测**仍成立。

10/2云端未获得9/30本地材料；其“base Gate4未测”不能用来否认9/27独立诊断，也不能拿独立33步的0/5与trained完整500步Gate4作匹配成功率比较。本次没有复测base。

## 3. 目标与执行为什么可脱节

旧Qwen：`reports/training/m2-qwen08b-frozen-001.md:205` 的validation MAE `0.0230362`，较train-action-mean降低87.01%；严格300步×5的Gate4仍0/5、reward0（同文件257，及 `reports/training/m2-qwen08b-frozen-001-model-card-addendum.md:34`）。旧100步/碰撞分类记录是历史阶段，不能混入最终验收。该系统是 frozen pooled Qwen feature + Rosetta action head，`scripts/train_m2.py:110` 调用 `src/rosetta_reality/train/losses.py:12` 的SmoothL1 action目标；不是Qwen ER独立能力验收。

SmolVLA：`reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md:26` 及其JSON记录MAE改善73.81%、Gate4 0/5；`reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md:26`、57、86记录两臂first-action MAE较base改善约92%，但都0/5。这足以确认同类**指标—行为脱节现象**，不用假定Qwen与SmolVLA使用同一目标或有同一坏模块。

SmolVLA原生目标是归一化动作空间的flow velocity MSE，经有效target/mask reduction；不是插入成功、碰撞约束或多步闭环代价。架构入口 `docs/m2-smolvla-architecture.md:1604`、`src/rosetta_reality/vla/horizon_loss.py:114` 区分未归约flow loss与最终reduction。ABC保留native loss，`src/rosetta_reality/vla/training/paired_mixture_runtime.py:255` 只包装finite检测，未将目标变成success优化；仓库SmoothL1函数不在这条native训练路径上。本次没有读取容器内上游实际模块，不对其实际loss denominator给出新的运行认证。

具体脱节候选：

1. 训练/离线评估观察teacher轨迹，执行观察学生自身诱导的状态。偏离后按旧时刻动作评MAE，不等于该偏离状态的正确恢复动作。
2. 50槽uniform目标与50/1执行权重不同，slot0误差可能被其余槽稀释；**50/1不是已证实实现bug**。Zen firstaction也0/5，故“只改first-action weighting即可修复”已有反证，不能从Faust旧待办直接再启动。
3. normalization改变维度权重；臂rad、夹爪normalized不可把一个聚合MAE当抓取/插入误差。bounded-sine映射合法性也不证明内部latent或开合时机正确。
4. teacher的几何/接触/历史比学生单帧输入丰富；可能涉及部分可观测性或不同阶段相似观测的动作冲突，但还没有匹配输入与不同标签的冲突实测。
5. zero-noise reload只认证一个固定输入的完整chunk相等。Gaussian rollout每次推进专用CPU RNG再转dtype；相同seed不等于实际张量/调用次数已核验。canonical旧同step5000跨run差异已有9/24分析:89，不能本次重复记为新模型复测。

ABC A20k D、B/C15k D+5k R/S，辅助仅一个parent、扰动后起点曝光6.05%，valid slots也不相同；这是9/30已有发现，本次不冒称新预算核验。三臂训练loss分别来自不同分布，不能排序恢复能力；它们也没有提供本次共同验证集MAE改善证据。

## 4. normalization / 时间 / 合同一致性与尚未排除项

| 边界 | 当前证据 | 限制 |
| --- | --- | --- |
| 单位/ordering/时间 | `configs/sim/aloha_insertion_smolvla.yaml:10`：14维absolute joint targets、50Hz、obs_t→t..t+49、执行slot0；夹爪维6/13为0闭1开 | 本次未核验上游actuator实际映射和完整物理步状态 |
| representation | `src/rosetta_reality/vla/processor.py:223`、287：projection后pi-Aloha/夹爪映射，normalization后模型，unnormalization后逆映射 | 流程声明与保存文件匹配不等于逐次有效processor运行值 |
| 训练processor | `scripts/smolvla_forward_check.py:204`覆盖features/stats/rename | ABC历史CPU输入检查存在；本次未执行processor，不代替CUDA运行证据 |
| 执行processor | `scripts/smolvla_autodl_vfunfreeze_sim_gate.py:259`覆盖normalizer/unnormalizer和rename、302确保边界 | 6维placeholder vs14维dataset并不能单独定bug；有效steps/features/stats和值仍未采集 |
| bank解码→命令 | `scripts/evaluate_paired_policy_v2.py:183`执行clip(raw[0])，并非直接执行返回projected | 三条2445 raw_clips=0；缺完整raw/projected，未证两条路径在bank逐值相等 |
| command→physics | `src/rosetta_reality/sim/gym_aloha.py:254`执行logical action，再查实际qpos/contacts | 合法命令不保证物理不越限；缺ctrl/qvel/joint margin，不能定位全部动力学责任 |

已排除的范围仅包括：151文件接收后SHA漂移、14登记evaluation源码漂移、15保存processor/config漂移、2445首槽需裁剪、这三条未执行任何物理步就拒绝、所有teacher源bank均无成功恢复证据。没有排除：有效processor不一致、恢复图像/状态不一致、Gaussian调用序列差异、actuator/动力学瞬态、teacher特权信息、恢复覆盖不足或模型在真实偏离输入下的动作误差。4条无stop却reward0的完整500步003场景是另一路任务失败，不能用安全停止解释全部0/5。

## 5. 检查结果、阻塞与最小下一步

执行命令（仓库根，PowerShell）：

```powershell
git branch --show-current
git rev-parse HEAD
git status --porcelain=v1 -uno
./scripts/investigate_paired_boundary_20261003.ps1
./tests/test_paired_boundary_20261003.ps1
git diff --check
```

最终调查脚本退出0：151测量SHA、14源码SHA、15processor/config SHA、8源bank SHA、3个post-step停止边界通过；测试退出0：**17项PowerShell静态/保存证据回归通过**（含source step/行号、pre-grasp、命令合法与物理stop区分、夹爪数值、有限证据范围、create-only保护）。`git diff --check`退出0。这是文件/数学检查，不是ML单测或新Gate。首版脚本因本地bank audit在collection的父目录而失败，未生成结果；只修本次调查脚本的本地路径，再核验plan SHA。没有移动旧文件或改runtime路径。

旧Python16项unittest **本次未重跑**：普通WSL调用返回`Wsl/Service/CreateInstance/E_ACCESSDENIED`；经执行环境授权的只读 `wsl.exe bash -lc 'docker image inspect sha256:4f91ff8e38a2ac6e3e0a307f335f82764c58abfedf15cebf5d67e17fe16b9fc5 --format "{{.Id}}"'` 返回退出1、`docker could not be found in this WSL 2 distro`。随后一次发行版只读查询遇到执行连接中断；项目PowerShell重新连接成功。没有擅自安装Docker/依赖、启动/重配服务、改系统环境或用Windows Python绕过合同。恢复已有WSL→Docker入口后，才可在9/30固定digest、network none、512MiB/1CPU、read-only条件重跑旧unittest；不需要先进行任何模型计算。

动态因果缺口不能从旧日志重建。提出如下**待用户决定、未授权执行**的最小诊断：

- 新identity与create-only输出；固定当前ABC step5000、revision/config/code/processor SHA、50/1、BF16、原安全阈值，optimizer=0，不换checkpoint或训练数据。
- 只选2445。在同runtime从完整保存branch恢复，先逐值记录integration/qpos/qvel/ctrl、Python/RNG、实时obs与保存obs差异。teacher正控制先重放源step178的保存命令一个物理步；这是命令可执行性正控制，不重新声称独立teacher成功。如需完整teacher恢复链，必须重建其原阶段/历史，另登记预算。
- 原入口A1/A2两次同噪声、相同初态各1个物理步；加observer入口A3同样1步；B/C各1步，共**6个物理步、5个policy forward**。先校验A1/A2，再校验A3；不得二次forward或多抽随机数“采集日志”。teacher命令正控制失败则停止学生能力归因。
- 采完整noise、preprocessor前后输入、实际resolved steps/features/stats、归一化完整chunk、raw/projected、首槽command、ctrl、全物理joint/value/range/margin与contact/object/EEF几何。输出最多256MiB；CPU内存上限8GiB、本地不得超16GiB总预算；若用已登记AutoDL容器GPU，暂建议allocated8GiB/reserved10GiB，单worker，加载+采集15分钟硬停止。这是保守上限草案，性能/VRAM未新测，不是费用报价；付费启动需另明确授权。
- **停止/转向**：身份不一致、缺字段、输入restore不同、A/A不同、observer改变noise/actions/states、nonfinite、预算超限或安全stop即停并保存失败。初态/processor异常先修工具链并复验同一对照；parity完整而学生首槽偏离安全teacher、物理状态变化一致时，才加强模型/目标候选；仍不直接重训。若2445解释只覆盖安全stop，下一登记应针对A1000无stop失败的抓取/保持几何，不扩到完整15场景。

本次没有改变组件入口、合同、控制流或Gate状态，因此没有再修改已有dirty架构文档。M2仍未完成；所有直接任务表现证据属于Gym-ALOHA/MuJoCo仿真，没有实机能力结论。
