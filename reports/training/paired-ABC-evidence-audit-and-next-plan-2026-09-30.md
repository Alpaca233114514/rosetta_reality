# Paired ABC 003：实际证据审计与下一步计划

日期：2026-09-30。性质：只读回传复算、源码审查、实验设计与离线工具支持；未启动训练、模型推理、仿真或远端计算。原测量与[原回传报告](paired-ABC-measurement-retrieval-2026-09-30.md)不改写。本报告不改变正式 Gate 状态。

## 1. 决策结论

先补齐初态、处理器、噪声和物理执行链，再选择单轴实验。目前不能把三臂首次动作差异当作首次错误，也不能由 B/C 停止较多推出恢复数据有害。三臂任务都未成功，安全停止造成观测时长不同；这批小样本不支持能力排序。

最有用的新证据是 **bank-003b-scene-2445 的 teacher 存在同安全实现下的成功恢复链**：原始 293 行记录没有物理越限、意外碰撞或裁剪，持续几何恢复之后完成任务；三臂 policy 接续该分支却都在第 1 步安全停止。它反驳“所有已测 bank 状态都没有可恢复证据”，但不能独自定位处理器、动作、恢复观测或执行动力学中哪一环出错。

本次已实现纯文件审计、上下文复算和可选证据 sidecar，并提供明确不可运行的诊断草案。sidecar **尚未接入实时 runner**，没有声称补回旧日志缺失的信息；实时边界接入与 observer parity 是下一阶段的必要工作。本次没有修改训练目标、控制器、安全限值或旧测量入口。

## 2. 独立核查范围与实际覆盖

测量根目录：`.work/paired-ABC-measurement-20260928-003-received-20260930-001/files`。重新计算 `remote.sha256` 的全部 151 个文件，检查清单与实际文件集合完全相等；核对 registration、evaluation plan、artifact manifest、episode summary/JSONL、bank summary/JSONL、参数不变收据及两个独立进程的 reload proof。三臂登记的 14 个源码文件与本地实际内容全部匹配。工作区身份为 `20260928T053233Z-66a5bbcdbdcd-9297dde2ac3b`；不以远端旧 commit 替代这些内容身份。

固定端点为第 5,000 个训练 update；本测量 `optimizer_updates=0`。`original_gate4_protocol_equivalent=false`、`formal_gate_result=null`；这是登记的探索测量，不是正式 Gate 4 验收失败或通过。

| 证据 | 五场景实际记录 | bank 实际记录 | 可得结论与缺口 |
|---|---|---|---|
| 逐步动作 | 4,187 步均有 14 维 raw、processed、executed | 200 步只有 executed | 五场景可逐值复算首槽裁剪；bank 无法复算 raw→projected |
| 逻辑状态 | 每步 state_before/after | 每步 state_after | 都不是全部 MuJoCo qpos；夹爪观测还涉及编码 |
| 物理安全 | 越限数量、意外 contact pair | 同左 | 可定位首次事件及接触名称，不能恢复每个越限 joint/value/range |
| reset/restore | reset 只有 seed、安全数量/pairs；首步含逻辑初态 | 没有实际 restore 初态行 | 不能证明当次完整 qpos/qvel、object pose、Python/RNG 状态和实际输入图像 |
| 任务几何 | 未采集 | grasp、peg/socket 接触布尔量 | 五场景 reward=0 不能细分未抓到、掉落、未对准等机制 |
| 噪声与全 chunk | 无 rollout 完整噪声、无完整预测 chunk | 同左 | seed/生成代码不等于实际张量证据 |
| 处理器 | manifest 有文件 SHA，无逐次生效配置 | 同左 | 保存身份可验证，运行时覆盖与实际输入 shape/dtype 未直接采集 |
| 实际控制量 | 无 actuator ctrl | 无 actuator ctrl | logical command 不等于电机实际控制量或测得 qpos |

五场景共对每条动作链的 58,618 个维度逐值检查：`processed == float32边界裁剪(raw)`，`executed == processed`，相邻逻辑状态连续；全部通过。这里 raw 已是后处理逆变换后的物理动作空间，不是网络归一化输出。bank 共 9+106+85=200 步，没有将缺失的 raw/processor/physics 字段推定为存在。

`diagnostic_snapshot` 在现有 adapter 中能提供更多信息，但旧 runner 没调用，故“可采字段”不能计入实际覆盖。完整覆盖计数、首个事件原行和末行见[测量复算 JSON](paired-ABC-evidence-audit-2026-09-30.json)。

## 3. 碰撞、物理越限及未安全停止的失败

以下步号均为 policy 执行步，1 起算。`L10-L/R` 分别指 `vx300s_left/10_left_gripper_finger` 与 `vx300s_left/10_right_gripper_finger`。

| Arm | seed | 首次物理碰撞 | 首次物理越限 | 执行步数/结果 |
|---|---:|---|---|---|
| A | 1000 | 无 | 无 | 500，任务失败 |
| A | 1001 | 434，table / L10-L | 无 | 434，安全停止 |
| A | 1002 | 74，table / L10-L | 无 | 74，安全停止 |
| A | 1003 | 61，table / L10-R | 无 | 61，安全停止 |
| A | 1004 | 无 | 无 | 500，任务失败 |
| B | 1000 | 无 | 125，数量 1 | 125，安全停止 |
| B | 1001 | 无 | 无 | 500，任务失败 |
| B | 1002 | 无 | 128，数量 1 | 128，安全停止 |
| B | 1003 | 无 | 115，数量 1 | 115，安全停止 |
| B | 1004 | 125，table / L10-R | 无 | 125，安全停止 |
| C | 1000 | 无 | 无 | 500，任务失败 |
| C | 1001 | 无 | 499，数量 1 | 499，安全停止 |
| C | 1002 | 无 | 128，数量 1 | 128，安全停止 |
| C | 1003 | 无 | 115，数量 1 | 115，安全停止 |
| C | 1004 | 无 | 383，数量 1 | 383，安全停止 |

A/B/C 均 0/5；安全停止为 3/5、4/5、4/5；累计步数为 1,569、993、1,625；碰撞计数为 3、1、0；物理越限计数为 0、3、4。所有 15 条 maximum_reward=0。未安全停止的四条恰为 A1000、A1004、B1001、C1000，均耗尽 500 步预算；没有几何轨迹或图像证据足以判定具体任务失败阶段。

安全实现见 `src/rosetta_reality/sim/gym_aloha.py`。物理越限检查遍历 `jnt_limited`，以 `jnt_qposadr` 取实际 qpos，比较模型 `jnt_range`：`qpos < lower - 1e-5` 或 `qpos > upper + 1e-5`。旋转/平移关节的单位由模型决定；不能拿 14 维动作裁剪边界替代该检查。接触检测基于 MuJoCo contacts 与允许接触规则；机器人碰桌属于 unexpected，没有另设接触力/距离阈值。本次不放宽这些规则。

唯一 raw 裁剪发生在 C1003 第 114 步，`right_forearm_roll`：raw=3.143169641494751 → projected=executed=3.141580104827881（float32 上界），逻辑 state_after=3.1362297534942627。第 115 步 raw=projected=executed=3.126471996307373，但逻辑 state_after=3.142620325088501，出现物理越限数量 1。B1003 第 115 步相应逻辑观测为 3.144552707672119，命令为 3.12207293510437。命令合法而测量越界与这些记录相容；右前臂是具体嫌疑维度，但日志缺完整 qpos/joint 明细，不能宣称已证明唯一触发关节或“裁剪引发了下一步越限”。其余越限事件同样不能从汇总计数恢复 joint 名称。

## 4. 恢复 bank 的可恢复性与初态证据

注册状态为 2440–2449；2442、2448 没创建 branch，因此每臂测量分母为 8，不把它们计为 policy 失败。下表 `L/C` 为首次安全停止时的越限关节数量/意外接触条数；每个已测状态首次安全事件就是终止事件。

| state | A：step L/C | B：step L/C | C：step L/C |
|---:|---|---|---|
| 2440 | 1，2/0 | 1，1/1 | 1，1/2 |
| 2441 | 2，2/0 | 1，1/2 | 1，1/2 |
| 2443 | 1，4/0 | 1，3/0 | 1，3/0 |
| 2444 | 1，2/0 | 1，1/0 | 1，1/0 |
| 2445 | 1，2/0 | 1，0/2 | 1，0/2 |
| 2446 | 1，2/0 | 65，0/2 | 3，0/1 |
| 2447 | 1，2/0 | 35，1/0 | 76，0/2 |
| 2449 | 1，3/0 | 1，1/0 | 1，2/0 |

B2440 是右臂 `10_right_gripper_finger` / table；B2441/2445/2446 及 C2440/2441/2445/2447 是右臂两个指体 / table；C2446 是左臂 `10_right_gripper_finger` / table。A 没有 contact 触发。逐条原始名称保存在复算 JSON。三臂 task_success、持续几何 recovery、verified recovery chain 都是 0/8；C2447 maximum_reward=2，仍未满足恢复链条件，不能改记为成功。

源 bank 位于 `.work/paired-bank-v3-20260927-001-independent-received/extracted/collection`。manifest SHA=`3fe172d5504b66a378d854b02c1607a332f5f0c0df1e18c5a5b5e56c26bac69c`，collection-audit SHA=`dc1a8a0179f920a33588baa0b546390b647e3579ee27b4f4d762498c75c544b6`，均与 003 登记相符。检查了源 labels、summary、branch integration/observation/Python receipt 的绑定；source receipt 的 safety adapter 与本地/003 相同。

按源记录的 0 起算步号复算：2445 correction_start=178，持续几何 recovery=190，terminal=292，完整 293 行安全；2440/2441/2444/2449 分别在 193/188/219/191 达到几何恢复，但预算内无 terminal；2443/2446/2447 的偏离没有验证、无持续几何恢复，即使有 terminal 也不能当成功恢复链。2442 未生成，2448 在源采集出现碰撞。不同状态须分层报告，不能把 teacher_feasible 标签直接等同“同协议可恢复”。

2445 偏离后源逻辑右前臂为 2.793635845184326；A 首条命令该维为 0.1134498193860054，相差 2.6801860257983208 rad。B/C 首条命令最大的臂关节变化在左 wrist_rotate，约 0.421/0.423 rad。这是值得定向检查的命令跳变，不是动作错误的 ground truth；缺实际 ctrl、全关节状态和当次恢复输入，不能推断纯模型原因。

原 branch 已有完整 integration 与 Python/RNG receipt，restore 代码会校验源/model/spec、恢复并校验 integration/Python 状态、返回缓存 observation。**源状态可追溯与当次 rollout 完整初态已落盘是两件事**：003 没保存 restore 后、首个动作前的完整边界。当前只复核已有文件，没有重放仿真。详见[上下文复算 JSON](paired-ABC-context-audit-2026-09-30.json)。

## 5. 处理器、噪声与预算核对

三臂本地 step5000 checkpoint 的 pre/postprocessor JSON 及两个 stats safetensors 共 12 个文件均与 003 artifact manifest 匹配；另核对 3 个 checkpoint config。共同 preprocessor SHA=`aa99381820bd8b7938956fe26a909aee76c88491a3cb0fb27f70d4748397c1da`，postprocessor SHA=`d0f0fef9507ca69e94f7f0ce9f1563878b85b27f5dac7c0952f7fb96a6133495`，stats SHA=`2021037779c22e8ab7c3344d9d8f7e5ce65a8c13326eab3201eb04eff2aff25d`。D normalization JSON SHA=`263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988`。

源码路径为 projection → pi-Aloha arms/bounded-sine gripper representation → STATE/ACTION MEAN_STD normalization，图像为 IDENTITY；postprocessing 逆向后输出物理空间动作。保存的 state feature shape=6 是上游遗留声明；artifact dataset 与实际逻辑输入为 14。`_validate_policy_contract_shape` 明确使用 dataset state dimension，未证明保存的 input feature 已被改为 14；runtime 又覆盖 normalizer 的 features/stats/rename 并确保动作边界。故不能由“6 vs 14”单独判 bug，也不能由文件哈希一致推定每次实际生效处理器已完整验证。下一次应记录 resolved steps/features/stats 与归一化前后输入形状及值。

源码噪声路径为 CPU generator、逐场景 manual_seed、float32 完整 `(1,chunk_size,max_action_dim)` 高斯噪声，再转到 CUDA/处理后状态 dtype；predict 前 reset policy。这解释了预期行为，不能代替实际噪声张量和调用序号。两个独立进程的 reload 在 seed=20260809、zero noise 条件下完整 50×14 raw/projected 相等且参数不变；其结论不外推为 seeded rollout、全场景或 sidecar observer parity。

实际预算来自 `source-bundle-20260927-003` 的 schedule/actual-updates 及训练 completion：每臂 5,000 次连续 successful_update、20,000 个登记样本曝光，公共 D 的 15,000 个调度项及逐 update 学习率序列精确相同。

| arm | 实际曝光 | 不同 source/episode/frame 输入数 | 已记录 valid slots | 已记录 padding slots |
|---|---|---:|---:|---:|
| A | 20,000 D | 16,227 | 951,354 | 48,646 |
| B | 15,000 D + 5,000 R | 15,203 | 963,303 | 36,697 |
| C | 15,000 D + 5,000 S | 15,203 | 963,303 | 36,697 |

曝光数是对 schedule 逐项计数并与执行汇总核对；valid/padding 是 runner 已记录的汇总值，缺逐批 mask，不能声称本次从 mask 独立重新求和。B/C 比 A 多 11,949 个 valid slots，约 1.26%；固定曝光预算并非相同有效目标数，原记录也没有声称两者相等。

辅助来源仅一个 parent 2436，`N_exploratory=1`、`N_strict=0`。203 个可采 chunk 起点中 154 个在共同前缀（0–153）、49 个在偏离后（204–252）；实际 5,000 个辅助曝光中 3,790 个前缀、1,210 个偏离后，后者占全臂样本曝光 6.05%，不等同去除 padding 后的有效目标比例。因此本轮不能检验“足够多样的 on-policy recovery 是否有效”，也不能以重采样次数当成独立恢复场景数。后续若改变覆盖，必须同时透明登记阶段曝光与有效目标计数，避免暗中改变比较目标。

## 6. 六篇论文对下一步的约束

本节对照官方全文；本地已有六份 PDF，无须重下载，也未改动其他任务的论文/Drive 同步目录。

| 论文 | 与本案相关的原则 | 本案下一步 |
|---|---|---|
| [SmolVLA，2506.01844](https://arxiv.org/html/2506.01844v1) §4.7 | prediction chunk 与 execution horizon 分开。论文专门比较 horizon=1/10/30/50；50 槽预测只执行 slot0 本身不是 bug。 | 保留当前 50/1；先证明动作/噪声/初态一致。若证据指向重规划噪声或节奏，再单独比较 horizon，不能连带改数据或训练。 |
| [ACT，2304.13705](https://arxiv.org/html/2304.13705v1) §IV | 命令目标与 follower 测得位置不同，差异影响 PID/接触行为；temporal ensembling 融合同一执行时刻的重叠预测。 | 保存 command→ctrl→qpos，不能用 measured qpos 替代命令正确性；不要把相邻动作平均冒充 ACT 集成。 |
| [RaC，2509.07953](https://arxiv.org/html/2509.07953v1) §4.2–4.4 | 在 policy 将失败的状态介入，恢复到已知分布，再完成子任务；recovery/correction 的来源与阶段重要。 | 优先采真实 policy 失败邻域且经安全 teacher 验证的状态；单 parent 人工腰关节偏离不能视为论文级覆盖。 |
| [IWR，2012.06733](https://arxiv.org/html/2012.06733) §IV | 干预与非干预样本重加权，论文使用两类等比例 batch。 | 当前 3D+1aux 不是 IWR 的直接复现。采样权重是独立实验轴，不能与场景扩充、horizon 同时变化。 |
| [DAgger，Ross 等，2011](https://proceedings.mlr.press/v15/ross11a.html) | 在 learner 实际诱导的状态上取得 expert 标注并聚合，处理分布漂移。 | 偏离后的时间对齐参考动作不自动是真实 expert 标签；需要状态条件 teacher/干预证据，而非只有旧轨迹时刻标签。 |
| [Predictive Safety Filter，1812.05506](https://arxiv.org/html/1812.05506v4)，[Automatica 2021 DOI](https://doi.org/10.1016/j.automatica.2021.109597) | 预测约束、安全终端集合、备用控制器及模型误差假设构成安全保证。 | 当前 clip+事后 stop 不是预测安全证明。先采状态裕度与执行证据；新安全控制器须独立设计与验证，不在本补丁里放宽或替换安全合同。 |

论文的实验成功率不能直接外推给 Rosetta；此处用于确定比较轴与证据要求。

## 7. 分阶段实验计划（本次不执行）

**阶段 0，已完成：** 对旧文件做精确清单、动作链、安全/任务计数、来源身份、bank teacher 和预算核对。将完整性通过与因果证据不足分成两个字段，保留旧协议、缺失字段、未创建分支和负结果。

**阶段 1，先验证先验和采集：** 固定现有三个 5000-step artifact、optimizer=0、50/1、现有安全规则。先登记小型诊断 pilot：2445 同初态 teacher 正控制与 A/B/C 接续，A1003 碰撞，C1003 裁剪/越限，A1000 无 stop 失败。选择是有目的的诊断抽样，不能用其估计总体成功率。资源/磁盘/时限预算和执行授权另行登记。

采集必须在真实调用边界取得，不二次调用模型、不多走 physics、不额外抽随机数：reset/restore 后完整 integration、qpos/qvel/ctrl、object pose、Python/RNG receipt 与实际 observation；每次推理的完整噪声、调用序号、归一化输出、物理 raw/projected chunk、归一化前后输入及有效 processor 配置；每个物理步的执行槽、command、ctrl、全关节限值/裕度和接触/任务几何。图像或可取回的带 hash 引用必须留存，仅有 hash 不足以检查内容。

新 observer 接入后先做固定输入 A/A 与原入口/加日志入口的已登记一致性验证，核对实际动作、状态、噪声推进、停止步和参数；不以某一次 zero-noise reload 代替。采集缺失、身份漂移、非 finite、越过安全停止继续执行或超字节预算即停止诊断并保留不完整收据。通过条件是证据完整/一致，不是必须成功。扩大到完整 15 条场景与 24 条已创建 bank 接续应另有预算；2442/2448 仍为未测。

依次排查：①当次初态/输入是否与源 branch 一致；②归一化、camera rename、shape/dtype、逆变换是否一致；③首条命令及随后的 ctrl/qpos/接触变化；④四条无 stop 失败的抓取、保持、对准阶段。任何单个检查失败先修正并重新验证同一诊断；不要立刻重训。

**阶段 2，仅在阶段 1 通过后选一个轴：** 若主要是恢复状态覆盖不足，保持 native loss、初始化/归一化、batch/更新预算、50/1、teacher 与安全规则，登记更多独立、可恢复、policy 可达的 paired R/S parent，并分别计前缀/偏离/恢复/完成曝光。预先冻结 train/dev 按 parent 分组与保留评估状态，不能把当前 bank 诊断选择当独立测试集。若证据更指向执行节奏，则固定已训练 artifact，仅比较 execution horizon；明确按物理步还是重规划次数耦合噪声，否则比较被噪声变更混淆。两项不同时开展。

如后续采用历史 20k 曝光预算，明确估计的是固定更新/曝光替换效应；若要匹配 valid slots，须单独登记新的比较协议，不能悄悄改变历史预算。IWR 式权重改变另列一轮。任何新训练目标、硬件安全合同或预测安全控制器都需先提交具体设计，不能由本草案授权执行。

## 8. 代码与验证边界

- `scripts/audit_paired_measurement_files.py`：纯标准库，验证精确 SHA 清单、登记/summary 绑定、float32 裁剪链、动作/状态连续性、首个碰撞和越限、停止生命周期、bank 分母与持续恢复条件、reload 证据范围。输出 create-only，禁止写入旧接收根目录；缺字段明确计为缺失。
- `scripts/audit_paired_measurement_context.py`：只读核对 14 个源文件、bank 源 labels/branch 文件、训练 schedule/actual update/completion、processor/config 文件；不导入 ML/物理库，重要检查不用会被 `python -O` 去掉的 assert。
- `scripts/paired_measurement_evidence.py`：纯 Python sidecar 写入接口，验证完整张量 shape/dtype/values、物理 joint/value/range/阈值、非空 SHA 身份、身份漂移、首槽命令链、停止后禁止追加、create-only 与字节上限。输入是调用方提供的观测值；不能认证其真实来源，也没有自动采集钩子，不能声称已获得新实验数据。
- `configs/diagnostics/paired-measurement-evidence-draft-20260930-001.json`：不可运行、未授权执行的需求草案，明确接入/parity/资源预算待定；不是旧 protocol 的升级替换。

最终测试使用既有 Linux Docker 镜像 `sha256:4f91ff8e38a2ac6e3e0a307f335f82764c58abfedf15cebf5d67e17fe16b9fc5`，由 WSL 启动，禁止网络/拉镜像，512 MiB、1 CPU；只运行标准库单测和文件读取。测试结果见[验证记录](paired-ABC-evidence-validation-2026-09-30.md)。未测试：真实 processor/model forward、模拟器或真机、sidecar 的实时集成、observer parity、新训练/新评估效果、预测安全控制器。软件检查通过不会补足这些证据。
