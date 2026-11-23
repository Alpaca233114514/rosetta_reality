# 三项实验设计草稿（2026-09-27）

身份：`three-experiment-design-20260927-001`。状态：**draft；execution_authorized=false；不可调度**。本次只读仓库文档、配置与代码，并创建本报告及配套 JSON；没有启动模型、数据读取、ML 测试、训练、仿真、SSH 或下载。主任务的数据下载独立继续，本设计不控制下载进程。

用户最新约束优先：**“实验一只剩未训练模型的 gate4 没跑，其他原样。”** 因此 Exp1 只列未训练模型 Gate 4 待办；下述新增数据、采样与预算建议仅属于 Exp2/3，不回填或改造 Exp1。

## 1. 当前证据与可回答的问题

已完整阅读 `AGENTS.md`、`docs/m2-smolvla-architecture.md`、Faust 2026-08-12 与 Zen 2026-08-27 审计的 MD/JSON，以及最新 pre/post 分析及核验 JSON、canonical 004 与 traced 005 完成报告 MD/JSON。关键边界如下：

- 9月24日已保存的 pre/post 证据：base 在共同 ALOHA 接口下 Gate 3 失败（9 次关节限位累计、1 次意外接触累计），base Gate 4 未测；trained Gate 3 通过、Gate 4 为 0/5。此为历史事实，不用它覆盖用户关于当前待办的说明，也不假定存在新的 Gate 3 通过记录。
- canonical 完成过 5000 个成功更新、batch 4、20,000 个唯一训练输入各一次。Hestia 的 frame-0 设计不是 canonical 当前训练覆盖情况。
- 004、005、prepost 001 的同一 step-5000 权重产生不同轨迹/安全计数；跨环境、跨插桩运行不能直接归因训练或数据。旧 005 完整 trace 不包含状态条件专家偏离参照。
- 现有 ALOHA recovery manifest 要求通过的状态条件 teacher gate、来源与 seed 隔离及明确扰动合同。已有 teacher 候选闭合为负结果，不能把其接口存在、某个 seed 成功或旧 pending 当作 recovery 数据准入。

三项比较互补：Exp1 问“既定一次训练前后怎样”；Exp2 问“固定训练预算，把部分原数据采样换成 targeted recovery 是否有益”；Exp3 问“同量普通成功新增数据与 recovery 新增数据，哪种更有益”。Exp2 的改善不单独证明 recovery 优于普通扩充；该区别由 Exp3 检验。三个比较不能相加成统一因果分解。

## 2. Exp1：只补未训练模型 Gate 4，其他原样

保留 `reports/training/m2-smolvla-prepost-gate4-plan-2026-09-24.md` 的原模型、共同接口、processor/normalization、Action Contract、Gate engine、reset、seed、噪声、chunk/execution、阈值与训练后结果。这里的 untrained 指 pinned pretrained base 未做本项目任务训练，不是随机初始化，也不是已恢复的历史 actual step-0。

唯一待办为 **base Gate 4**：原五个 seeds 1000–1004，每个最多 500 步、零 optimizer updates；是否能够执行，须核对对应**现成** Gate 3 通过报告及其 artifact/code/simulation-plan 绑定。当前设计读取的文件只证明旧 Gate 3 失败，现成有效依赖标为 `pending_verification`。缺失时保持待核对，不新增 Gate 3 实验、接口修复、离线采集、重训、训练后 Gate 重跑或其他诊断轴；不降低门槛，也不把 base Gate 4 填成 0/5。

既有离线与20步安全证据可以原样作为限定范围的 training-effect 读出：该历史短程条件下 trained 安全计数更好，尚不能计算两端完整任务成功率之差。只有 base Gate 4 合法补齐且原比较身份仍成立，才能报告该固定端点对照的成功差；若环境绑定不一致，原样保留两份结果并明确无法作严格配对因果解释。BF16→FP32 存储/初始化差别仍是历史比较的解释边界，不在本计划中修复。

## 3. 数据资格：目前没有已确认可直接启动的同域 matched cohort

下载进度与最终收据以 `reports/training/targeted-dataset-staging-2026-09-27.md` 为准。本草稿形成时新增批次的最终完整性验收 pending；本轮没有解码视频或扫描数据行。数据集名称含 recovery、文件齐全或 Action Contract 相似，都不证明存在可用 correction 标签。

| 候选 | 当前可确认范围 | Exp2/3 中的角色与缺口 |
| --- | --- | --- |
| ALOHA human，`cc571a3c661df81b566dbfde3d5c1e85fcdf7884` | 现有任务的原始50-episode来源；原 split 为40 train/5 dev/5 hidden | 原数据 D 的首选；重新下载同 revision 不是新增成功轨迹，不能重复计 N；尚无逐事件 deviation→correction→success 资格清单 |
| ALOHA scripted，`8ab660912970111cbb26738b11458e6fc4a4aed1` | 旧下载源/目的校验通过；同任务候选 | 可审核普通成功 S；须确认可执行合同、真实结果和与 D/评估场景无重复；不能从 scripted 名称推定所有轨迹成功或有恢复 |
| REBOOT USB-A / RJ45 | USB-A revision `cfa3498a3982eb24554e88e77140247871eee3eb`；RJ45 `f6acd5b69394ffd4d2d8bd30079306c9e2061bbe` | WidowX follower、14维 joint/carriage、30 Hz；与当前 ALOHA14维绝对关节/50 Hz 不兼容。只作独立合同/标签审计候选，不拼入 D |
| RJ45 子集 | 元数据为24 episodes/21,559 frames，视频引用已入下载清单 | 元数据不是行数/解码验证；原全量 stats 只作 provenance，必须显式 subset view、split 与 train-only 重算 |
| Sirius、Can Paired、MimicGen square D0/D1/D2 | 已有各自来源身份/下载选择；合同分立 | 可研究干预/成功失败结构及未来独立任务对照；本轮未证明 recovery 对齐、同任务、同机器人或可用于 ALOHA Gate |

先做资格清单，后选 N；资格审计本身也不由本草稿授权执行：

1. 完整性：逐对象源哈希/大小与传后独立重读；episode/frame 数、时间戳、视频引用与解码对齐；许可与版本固定。保留原始 stats、失败传输和来源元数据。
2. 物理合同：逐维顺序/单位/坐标系、absolute/delta、控制模式、fps、state/action/image 时间对齐、gripper、chunk/padding、相机/任务语义。用原 Gate 1/2 证明可执行语义；不得以重采样30→50 Hz、改维数名称或张量同形代替机器人适配。
3. 标签：每条候选记录 `episode/scene/parent_trajectory`、原始片段范围、偏离事件、当前状态、专家接管/纠正起止、动作来源、恢复事件和最终 task outcome。缺失标签为 unknown；终帧成功不等于中间每步是专家 correction。
4. R 的合格条件：可复核的偏离→状态条件专家纠正→恢复/最终成功链，且覆盖偏离前后连续上下文；没有成功结局的 correction 单列 failed-recovery，不混作成功 R。仅有失败轨迹可作诊断，不能直接以原失败动作作正 BC 监督。偏离后的专家时间索引 replay 不是 recovery label。
5. S 的合格条件：可复核普通成功、相同任务/合同，排除显式恢复片段；没有介入字段不等于“无恢复”，无法判定则 unknown。同 episode 含普通/恢复段时全 episode 同 split，不把切片当独立轨迹。
6. 泄漏：按 parent trajectory/scene/采集来源分组后切分；去掉跨源重复与近重复，synthetic descendants 跟随 parent；验证/hidden/Gate场景及其纠正后代不得进入训练。先在 reader 入口排除原 hidden `[31,6,1,24,5]`；反复用于诊断的 dev 不再当盲测。

若同域 R 不足，则 Exp2/3 标记 blocked_on_cohort，不用跨机器人数据填足 N。后续需要另行授权的可能工作是：同一 ALOHA 场景来源采集普通成功及人工/已通过独立 gate 的 teacher recovery；或先建立另一 embodiment 的完整独立协议。两者都不属于本轮，也不重开旧 teacher 计划。

现有 `RecoveryDatasetManifest` 是 oracle 专用 v1 合同，外部人工纠正不能靠填写虚假的 `oracle_gate_status=passed` 准入；若有真实人工标签，需要独立版本的 provenance/资格适配设计及验证，保留现有 gate 安全基线。本草稿不实现该适配。

## 4. Exp2/3 的共同设计与唯一变化

记 D 为原 train40；S_N/R_N 为新资格清单选出的 N 条普通成功/成功恢复轨迹。N 是不同 trajectory 的数量，不是文件、切片或重采样次数。

| 新比较臂 | 训练数据支持 | 每 batch 4 的建议配额 | 比较 |
| --- | --- | --- | --- |
| A | D | 3 个共同 D slots + 1 个 D slot | Exp2 对照 |
| B | D + R_N | 同3个 D slots + 1 个 R slot | Exp2 treatment；Exp3 recovery |
| C | D + S_N | 同3个 D slots + 1 个 S slot | Exp3 普通成功 |

建议先审阅单训练 seed、三臂各5000成功 updates、batch4、accumulation1 的开发试验上限；均从同 pinned base 新初始化，固定最终5000，不做 checkpoint 搜索。该数值沿用 canonical 的可比较规模作为**预算建议，不是新执行承诺或本机可行性实测**。A/B/C 至多15,000主更新，独立两步 smoke 至多6更新且不得 warm-start 主臂；不得为了某臂不收敛而追加步数。Exp1完全不受这个新三臂方案影响。

共同 base slots 的 episode/frame/顺序逐一相同，15000次 D exposure；5000个辅助 slots 在 A 为 D、B 为 R、C 为 S。A 总 D exposure 为20000，B/C 各15000；**相同总updates、batch和加入R时，不可能同时保持A/B的全部D exposure相等**。因此 Exp2 估计固定算力下数据混合替代的效果，包含原数据曝光下降，不声称纯新增数量效应。Exp3 的 D exposure、辅助曝光和权重均相同，才较好隔离辅助数据内容。

可将 canonical 的已封存20,000-input顺序映射到上述 slots，但实际清单尚未构造。历史 trained artifact 仅在初始化、完整采样表、实际updates、processor/runtime与下列控制全部匹配时可作为 A；否则只是只读参考。需要新 A 时属于 Exp2/3 后续登记，不改动或重复 Exp1。B 能在 Exp2/3 共享，仅限完全相同 cohort、N、sampler、权重、训练与评估身份，不能算两次独立重复。

### N、有效帧和长度匹配

先按任务、机器人、场景来源/难度、起始位姿与采集者分层，再做 S/R 一对一 episode 匹配；有自然配对则优先。N ≤20作为首轮建议上限，实际 N 与 episode 清单为 null，待资格审计决定；不得按后续策略表现挑 N 或删难例。R 中纠正类型/严重度分层比例预先固定，匹配只定义适用子总体，不能消除所有选择偏差。

每对匹配持续时间与有效观测帧数；同时核对 `unique_input_frames`、有效监督 action slots、padding比例与重复exposure。采用 episode均衡、固定辅助slot配额，按完整阶段连续窗口匹配；每对有效输入帧及监督slots建议残差≤5%。无法达到则缩小双方匹配集或停止，不截掉失败、偏离或最终恢复事件来“凑齐”，不把重复帧算新增数据。报告各臂原始/保留/排除episode数、完整长度分布、有效帧、重复次数和加权质量。

S/R 若分别来自 scripted 与 human/另一个采集器，数据类型与采集风格相混；只能报告“这两个具体数据源的效果”，不能归因 recovery 机制。匹配时不按模型 loss、未来评估结果或训练后成功筛选。选择成功恢复R限定为“成功纠正示范的训练效果”，不能估计所有恢复尝试/失败的效果。

### 固定控制清单（仅 Exp2/3 新登记）

- 模型 `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`，pinned LeRobot `c903b114a90e703b3f7d0c46cb38727c328c55ff`；冻结VLM、相同expert/projection scope、native uniform loss及padding分母、不加augmentation/dropout/regularizer；实际初始化张量/buffer与dtype逐臂封存。
- 建议 seed `20260809`，AdamW LR `1e-4`、betas `(0.9,0.95)`、eps `1e-8`、decay `1e-10`、clip10，warmup16、cosine5000、终值`2.5e-6`，BF16；运行前核对真实LR序列、参数组、成功updates与scheduler步进。锁住实际完整noise/t与RNG策略，不以seed相同代替证据。
- 共用 D 的train-only normalization与相同processor；新增S/R先在各自train split重算stats作审计，保留source全量stats但不用于模型。主比较不分别采用S/R混合stats，以免引入第二轴；若固定D归一化不满足准入，停止并单独设计，不静默改归一化。
- 相同初始化、checkpoint/save/validation预算、固定终点、independent full-array reload；smoke/异常中断checkpoint不续用。相同camera/prompt/canonical pixels、14维Action Contract、50 Hz、50-action chunk、只执行slot0、不加history/temporal aggregation。
- Gate代码、MuJoCo/模型/渲染环境、reset、完整仿真积分状态、seed/noise、projection、collision whitelist、阈值、执行设备及精度固定。新观测器先经真实同环境A/A与透明性准入；合成测试或Basin导入不证明真实透明性，TorchLens不作为前置条件。

## 5. 读出、恢复评价与不确定性（Exp2/3）

主终点为原 Gate 4 的完整任务 success，保留 seeds1000–1004、500步和全部原安全标准；先通过各自 Gate 3。一个臂 Gate 3 失败则其 Gate 4 为 not_measured，不补零、不只分析幸存臂，也不能完成成功率优劣结论。最低0.2成功阈值加其他全部准则才构成既有Gate验收；1/5不代表广泛鲁棒性。

恢复是独立关键次终点：另行登记最多10个可恢复、安全、与训练/teacher调参/Gate隔离的公共分支状态，每臂同一状态/噪声，最多150步。状态由预先固定的独立参照与事件规则选取，不由某一治疗臂的表现选取；teacher需先证明这些状态可恢复。当前 bank、恢复事件规则和恢复阈值均 pending，不能调用旧失败teacher代替。若没有资格合格bank，报告 recovery not_measured，不用reward上升代替。每次应记录短暂恢复、持续恢复与最终任务成功，分母包括全部预登记事件及refusal/timeout/safety abort，基础设施缺测另报，不按事后“可恢复”删例。自然rollout的恢复率仅作描述，其访问状态因policy不同而不可直接因果比较。

分层读出至少包括：

| 层次 | 指标及边界 |
| --- | --- |
| 第一步 | 同reset/同完整噪声的首动作误差与post-state偏移；仅在该状态有合法专家参照时称expert deviation，否则称臂间差异。动作MAE按关节rad与夹爪normalized分组 |
| 中途/分支 | reset→raw输入→processor/noise→输出→executed action→下一物理状态的首差异；记录object/EEF/contact/reward及分支前后窗口。缺事件只报观测区间；不把策略分叉本身叫错误 |
| 20-step | 原Gate3安全/合法性、进度、首差异；20步未成功不等于完整任务失败。独立恢复bank的20步前缀另报，不能冒充原Gate3 |
| 恢复 | 固定bank内recoveries/全部事件、time-to-recovery、refusal/timeout、安全中止与最终success；恢复阈值来源在执行前冻结 |
| 完整任务 | successes/5、每seed结果、time-to-success/截尾、rollout length及全部Gate criteria |
| 离线 | 同dev、相同Gaussian seeds与zero-noise独立栏；first-action/full-chunk、关节/夹爪、阶段、padding、常数基线。dev只报告，不改固定终点或采样；不替代闭环 |
| 行为与资源 | finite/raw/internal-support/projected/executed legality、限位与碰撞累计和事件数分开、smoothness、推理/仿真p50/p95及warmup/采集开销 |

首轮只是一组seed配对开发实验，training variance未估计。成功率报告精确计数与95%二项区间作为有限样本描述；paired success差报告discordant pairs及配对区间，连续指标按episode/scene聚合，不拿2500帧或多个noise当独立样本。固定5个已反复使用的开发seed不是随机部署样本；区间不能外推真实部署。两项数据对比共享B，报告其相关性；若以后做显著性决策，预先登记两项比较的多重性处理，不边看结果边加seed。更广泛结论需要另批独立训练seeds和未触碰评价场景，本草稿不授权。

“改善”须给出效应大小/区间和安全指标，不能由离线下降或训练loss认定；“M2通过”仅由原Gate全部准则及artifact/reload链决定。对于小样本不确定或安全退化，保留mixed/inconclusive，不能为得到正结论改成功定义。

## 6. 分阶段预算与停止条件

当前阶段只有静态设计，两文件以外不写。以下均为未来授权后可登记的阶段，任何缺失值必须先填实测/来源，不能仅把本JSON的布尔值改成true作为启动计划：

| 阶段 | 限定输出/预算建议 | 进入下一阶段的条件 |
| --- | --- | --- |
| Q：数据资格，只读诊断 | 一次分源contract/标签/split审计；首轮仅同域ALOHA；逐episode流式处理，禁止全数据进内存 | 最终下载验收、scene/trajectory清单、N/帧/长度匹配与teacher/人工标签依据全部明确 |
| P：实施与准入 | 固定容器、doctor、相关单测、Gate1/2身份；每臂batch1无更新forward与最多2个独立smoke更新、full-array reload | 无hidden泄漏、无参数/processor漂移；真实sampler/有效loss分母、A/A与采集透明性、资源通过 |
| T：科学数据比较 | 至多3臂×5000主updates；顺序运行、不并发GPU、无自动retry/resume/加炉；每臂建议工作封顶7200秒 | 每臂同成功update计数、LR轨迹、实际exposure/weights、完整checkpoint及恢复备份；任一未完成则不作完整同预算比较 |
| E：评价 | 每臂原Gate3最多20步；通过后原Gate4最多2500步；合格恢复bank最多1500步/臂，另行批准；离线清单/forward总上限需先封存 | 独立reload、Gate同源绑定、完整逐步证据/receipt/独立算术复核；按结果关闭，不自动下一炉 |

资源建议保留历史受限值：本地整机16 GiB设计、ML仅WSL Bash→Linux Docker；获批AutoDL用实测profile且不嵌套Docker。训练候选进程RSS≤8 GiB、CUDA allocated≤8 GiB/reserved≤10 GiB；这些是待准入的上限，不是新数据下的实测需要。当前下载容器曾有2 GiB cgroup限制，不能把主机内存或历史GPU环境当当前可行性证明。所有GPU阶段的实际wall预算、watchdog截止、证据字节/存储预算须依据新benchmark填写，当前为null；超过建议cap先停止，不自动扩大。

下载选择占66,158,775,113 bytes、预留10,000,000,000 bytes是下载规划，不是训练checkpoint可用空间。训练前重新实测free space并计算所有未备份checkpoint/optimizer、临时保存、日志/trace/video最大值；容量不足保留数据，另议备份/存储，不删除历史材料或把多臂未备份产物同时塞进余量。

共同停止条件：身份漂移、标签无法追溯、split泄漏、相同维数但合同不兼容、失配超过预登记界限、nonfinite/OOM、实际update或loss分母/采样不符、未授权skip/resume、A/A不一致导致归因无效、资源/截止/磁盘超限、Gate前置失败。失败证据保留，新身份修订；失败不自动授权新模型、更多数据、teacher开发或参数轴。

只读诊断回答“数据/证据够不够、比较能否成立”；训练B/C属于科学干预。发现普通工程问题应独立修复并验证后重新封存，不能在数据比较中顺手改dtype、optimizer、loss、adapter或Gate。

## 7. 尚待补齐与引用

尚缺：Exp1现成有效Gate3绑定；新增批次最终收据；同域S/R逐episode资格、匹配N与有效帧；独立恢复bank/恢复阈值；Exp2/3完整采样表/初始化/代码/processor/environment hash；当前资源benchmark、磁盘与逐阶段wall/字节预算；真实模型观察器透明性。本设计不伪造这些hash或把历史值标为本次实测。

主要当前来源（均为仓库文件）：

- `reports/training/m2-smolvla-prepost-analysis-and-next-plan-2026-09-24.md` 与 `m2-smolvla-prepost-analysis-verification-2026-09-24.json`。
- `reports/training/m2-smolvla-prepost-gate4-plan-2026-09-24.md`。
- `reports/training/m2-smolvla-canonical-fullframes-plan-2026-09-14.md`、`m2-smolvla-canonical-fullframes-gate34-result-2026-09-15.{md,json}`、`m2-smolvla-traced-gate-result-2026-09-16.{md,json}`。
- `reports/training/m2-smolvla-t2-geometric-teacher-003-closure-2026-09-03.{md,json}`；该报告MD/JSON的成功计数表述不一致，因此本稿仅使用两者一致的teacher gate未通过事实，不据此批准标签。
- `configs/sim/aloha_insertion_smolvla.yaml`、`configs/diagnostics/dual_axis_split_20260924.json`、`src/rosetta_reality/data/recovery_manifest.py`、`docs/dual-axis-diagnostics.md`。

本次不改变架构、Gate状态或旧实验授权；配套 `configs/diagnostics/three-experiment-design-20260927-001.json` 是设计记录，不是任何launcher的可执行run配置。
