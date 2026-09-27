# SmolVLA 桌面资料综合分析与下一步计划

本轮结论：**训练后的策略在本次短程安全检查中优于 base，但仍未学成可通过任务验收的闭环策略。下一步先补可区分假设的证据，再决定训练轴。** 没有依据认定“训练破坏了原本能完成任务的模型”，也没有依据把 33 个不变 norm 直接定为唯一根因。

分析身份：`prepost-desktop-analysis-20260924-001`。资料包名称为 `SmolVLA-PrePost-Gate4-Analysis-2026-09-24`；本报告使用仓库相对源路径引用证据。此次只读分析已保存的文件，不启动模型、数据加载、仿真、SSH 或训练。

## 1. 本轮实际核验

独立机器读出见 [analysis-verification JSON](m2-smolvla-prepost-analysis-verification-2026-09-24.json)。

| 核对项 | 本轮实测 | 证明范围 |
| --- | --- | --- |
| 桌面 manifest 清单 | 93/93 文件，2,534,531 字节；大小、SHA256 均一致；仓库源文件也全部一致 | 副本与已有声明及当前源文件一致；不是上游签名认证 |
| 原生 Gate 绑定 | 三份 Gate 报告均匹配其 YAML 与 artifact manifest；训练后 Gate 4 正确引用 Gate 3，代码身份一致 | 文件间身份链成立；包内不含链接的模型／processor payload，未重验这些大文件 |
| Gate 4 五个 episode | 逐回合 JSON 与汇总内嵌 metrics 相同；重算全部 12 个 aggregate 字段和 8 个验收条件一致 | 从 episode 摘要独立汇总；不能代替逐步物理状态复算 |
| reload 探针 | 每臂两份探针 JSON 逐字节一致；各含 raw/projected 的 50×14 动作 | 一个 reset 观测、zero noise 的保存证据一致；未重新执行模型 |
| 参数比较 | 三组各 500 行，共 1,500 条记录，名称覆盖、有限性声明、行文件 SHA 与总计一致 | 本轮重算 JSONL 统计；没有重复历史全权重数值计算 |
| Basin history | 四条 history 的 12 个 manifest 成员 SHA 一致，内嵌源报告与原生 JSON 一致；事件数为 1/0/1/5 | 同一实验证据的派生导入，不增加独立重复实验数 |

资料包 manifest SHA256：`b9fe5644c7d5938effc0274f117df08fa8be8311ae8560da1b99b609d99217ef`。README 和 manifest 本身不属于这 93 项成员。核验脚本为 `.work/analyze-desktop-prepost-20260924-001.ps1`，其执行版本 SHA 保存在 JSON 中。

## 2. 实验结果应如何解读

### 2.1 训练前后：短程安全有改善，任务成功仍未建立

| 项目 | base + 保存的 ALOHA 接口 | canonical step-5000 + 同一接口 |
| --- | ---: | ---: |
| Gate 3：seed 20260809，20 步 | 未通过 | 通过 |
| Gate 3 关节限位累计计数 | 9 | 0 |
| Gate 3 意外接触累计计数 | 1（左右手指互碰） | 0 |
| Gate 3 非有限／命令越界率 | 0 | 0 |
| Gate 4 | **未测量**，Gate 3 前置条件失败 | **0/5**，五回合最大 reward 均为 0 |

这支持“该接口与固定短程条件下，训练后端点的安全行为更好”。它不能外推为一般安全性提升，也不能计算训练前后 Gate 4 成功率之差。base 不是原始默认 6-D 接口，更不是保存过的历史 step-0；两端点也有 BF16→FP32 存储变化，因此本次对照衡量端点行为，不隔离优化器机制。

本轮训练后 Gate 4 的逐回合读数：

| seed | 步数 | 最大 reward | 成功 | 关节限位累计 | 意外接触累计 |
| --- | ---: | ---: | --- | ---: | ---: |
| 1000 | 500 | 0 | false | 0 | 0 |
| 1001 | 500 | 0 | false | 1 | 2 |
| 1002 | 500 | 0 | false | 0 | 16 |
| 1003 | 500 | 0 | false | 18 | 14 |
| 1004 | 500 | 0 | false | 0 | 17 |
| 合计 | 2500 | — | 0/5 | 19 | 49 |

49 项接触直方图全部是手指碰桌面：左手两指 13/8，右手两指 3/25。限位计数按每步违规关节累计，接触计数按每步意外接触对累计；均不是独立事故数或有事故的唯一帧数。19 项限位中的 18 项位于 seed1003；当前包没有逐步关节身份，不能把旧 005 的“全部是手指关节”结论移植到本轮。

**seed1000 没有限位或意外接触，仍完全失败。** 因而，仅降低这些安全计数不足以解释或修复全部任务失败。动作合同内的命令也不保证物理关节与接触安全，更不保证接近、抓取或插入成功。

本轮缺少物体／手掌轨迹、预处理输入、完整采样噪声及逐步奖励记录，不能仅由 reward=0 宣称“从未接触任何物体”，也不能确定失败首先发生在接近还是抓取保持。旧 005 的轨迹已定位到抓取建立／保持问题，但只直接证明旧 005 的行为。

来源：`reports/training/m2-smolvla-prepost-gate4-{plan,retrieval}-2026-09-24.md`，以及 `runs/prepost-gate-received-20260924-001/prepost-gate4-20260924-001/results/` 内原生报告。

### 2.2 权重确实学到了变化；不变 norm 是可检验线索

三个端点都有 500 个保存张量、450,046,176 个值。每对均有 122 个数值变化张量：112 个 expert 张量、10 个 state/action/time projection 张量；345 个 VLM 张量数值一致。base 到训练端点的另 199 项字节差异仅来自 BF16→FP32 的精确数值转换。

因此不能再把“整个模型没训练”或“所有视觉权重被训练损坏”当作工作结论。冻结 VLM 并不意味着下游视觉利用路径不变，也不代表视觉信息一定利用充分。

33 个 expert norm 在三端点均为 BF16 的全 1 张量。已有 CPU 标量反例证明：非零梯度和非零 AdamW 矩可以与 BF16 权重零有效变化同时出现。但它使用 CPU/torch 2.11.0+xpu；历史 CUDA 使用 torch 2.8.0+cu128。当前仍缺少：

1. 真实参数名到 optimizer 参数 ID 的可核验映射，以及这些 norm 实际是否在更新组；
2. 实际 norm 的存储／计算／梯度／矩状态 dtype，是否存在 master weights；
3. 固定真实训练输入上的 grad=None、真零、非零梯度和预期更新量；
4. 有限精度更新受阻是否会对行为产生有意义的影响。

即使确认更新受阻，也只能证明一个优化机制限制，不能自动证明它造成 Gate 4 失败。相对 L2 最大的 time projection／K/Q 层同样不是因果排名。

来源：`m2-smolvla-prepost-parameters-result-2026-09-23.md`、`m2-smolvla-prepost-bf16-scalar-probe-2026-09-23.md`、`m2-smolvla-prepost-remote-preflight-2026-09-23.md`（均位于本报告目录）。

### 2.3 同一权重的不同闭环结果，需要先控制复现条件

| run | 观测／环境区别 | Gate 4 | 限位／意外接触 | seed1002 最大 reward |
| --- | --- | --- | --- | ---: |
| canonical 004，9月15日 | 原运行环境，无完整逐步 trace | 0/5 | 19 / 38 | 0 |
| canonical 005，9月16日 | 原运行环境，有完整 trace | 0/5 | 27 / 60 | 2 |
| prepost 001，9月24日 | 新 clone，报告 GPU 为 RTX 4080 SUPER，无完整逐步 trace | 0/5 | 19 / 49 | 0 |

三者主权重 SHA 相同，说明这些变化不是另训了一个 checkpoint。环境、实际输入／噪声、采集方式和运行状态未全部配对，因此也不能说“trace 导致改善”“clone 更差”或“训练退化”。相同 seed 与 sampled reload 不能替代完整闭环 A/A。

004 缺少的历史轨迹无法通过重跑补回。下一次应建立一个新身份的同环境参照，不追求伪造旧 004 的内部路径。来源：本资料包 canonical 004 报告，以及仓库 `m2-smolvla-traced-gate-result-2026-09-16.md`、`docs/m2-smolvla-closed-loop-reproducibility.md`。

### 2.4 Basin 可用于查证；TorchLens 还不是真实模型诊断主路径

Basin 的四条 history 完整保留了“base Gate 4 未测量”。它们来自本次原生报告，没有增加新的模型运行、激活或梯度信息。`verify` 通过说明记录完整性，不是独立物理验收。

TorchLens 的合成 bridge 已经过测试，但 strict transparency 仍失败：Python RNG 改变，Python-random 参与计算时输出也改变；图完整性尚未独立验证。下一阶段以 plain A/A 与透明 module hooks 为数值参照。TorchLens 只在独立、限量、另行验收的离线进程中研究，不作为诊断主线的前置依赖。

### 2.5 历史诊断已经排除了一些简单方案

canonical 已消费 20,000 个唯一训练输入，不能把 Hestia 的 frame-0 小样本设计混作 canonical 当前的覆盖缺陷。9月22日的已有数组分析发现：frame125 左夹爪模型优于简单坐标／状态三近邻及常数，但有局部后期开发退化；frame250 保持幅值仍输训练常数。它们分别涉及时机、保持和泛化，不支持立即改标签或统一时间平移。

旧 first-action weighting、state jitter/dropout、vision unfreeze、Iris K 正则各有已完成的负结果与适用边界。不要将早期 Faust/Zen 报告里的“下一步”直接作为新的执行队列。参见 `m2-smolvla-root-diagnosis-{result,followup}-2026-09-22.md` 与架构入口的后续记录。

## 3. 下一步执行顺序

以下是**研究设计与工作排序，尚不可调度执行**。本次请求授权分析与计划；没有新建真实计算、训练或 rollout 授权。每阶段使用新的 create-only identity；实际 SHA、环境、预算、截止与停止条件在执行前固定。

### P0：完成本地准备，让一次计算回答一个明确问题

本轮已完成副本／身份／摘要算术核验。下一段本地工作应在现有功能分支上：

- 将 33 个 norm、155 个预期可训练参数、optimizer 序号／状态文件、初始化／dtype 转换路径列成显式合同；不按 shape 猜测 optimizer 映射。
- 对照既有 seed1002 三臂草稿和当前 source，确认没有过时 SHA、旧机器身份或已完成任务被再次排队。
- 准备 raw input、processor、完整 noise／timestep、loss denominator、RNG、参数／buffer 的保存与配对校验；hidden 排除必须在数据读取前生效。
- 先做不加载模型的 schema／静态检查，再按仓库要求在固定 Linux Docker 中做必要的合成反例检查。合成通过仅证明工具路径。

交付：一份可审阅的 norm 诊断计划、一份新的三臂配对计划，以及各自的 source/config/input identity 清单。不得提前开 GPU 等待本地实现。

### P1：优先做同环境 seed1002 配对复现，锁定行为证据

科学问题：同一模型的逐步差异首先出现在哪个边界，增加完整 trace 是否有可测影响？

- 固定 step-5000、保存 processor、同一个 GPU／库／渲染环境、seed1002 和完整噪声策略；不改 BF16、确定性开关、动作、控制频率或 Gate 阈值。
- 按既有工具定义执行 A1、A2（相同边界观察器）和 B（同观察器 + 旧 full trace）。每臂独立进程；先比 A1/A2，未匹配则停止对 trace 副作用的归因，B 可按预登记停止规则不执行。
- 先确认该身份下的 Gate 3 前置证据；最多三个 500 步诊断回合。它只诊断一个 seed，不能冒称新五 seed Gate 4。
- 保存 reset 与每步完整积分状态（含 warmstart）、obs／渲染身份、processor batch、实际 noise、normalized/internal/decoded/projected/executed action、物体与手掌位姿、接触和 reward。
- 比较顺序为 reset → 输入／噪声 → 模型输出 → 执行动作 → 下一物理状态。A/A 分叉时报告第一个不一致边界；不能只看末尾 reward。

验收：完整、可独立校验的配对证据，加首个分叉定位；若 A/A 与 A/B 均精确一致，才能把该采集配置用于后续轨迹机制分析。A 臂也有观察器，因此这仍不证明完全无插桩等价。

预算草案：先登记至多 20 步／臂的采集预检，按实测每步耗时与字节量决定是否进入长回合；含必要 Gate 3 在内，建议总工作上限 30 分钟、独立关机上限 35 分钟，证据上限沿用 4 GiB／臂、RSS 8 GiB、CUDA allocated/reserved 8/10 GiB。预计超限则在长回合前停止并修订预算，不默默提高上限。本轮普通 Gate 耗时不能直接预测带完整数组写盘的耗时。

### P2：无真实 optimizer step 的 norm 梯度与精度诊断

科学问题：33 个 norm 为什么没有端点净变化？此项准备可与 P1 的本地准备并行，但每次真实运行单独登记。

1. 先在受限离线 Docker 中只读解析既有 2500/5000 optimizer 状态，验证参数名／ID、moment dtype、step 与 native 参数组的完整映射；映射不唯一即停止该项推断。非零 moment 只能说明累积历史状态，不能重建所有历史梯度。
2. 在获批 CUDA 窗口，以固定 train-only 清单、batch1、同输入／完整 noise／t、target／mask 比较 base（共同接口）、2500、5000。建议先用 1 个样本通过 plain A/A 与 hook 透明性，再扩到 4 个固定样本，总计不超过 24 次监督 forward/backward。样本按预先固定的帧／阶段规则选取，不按结果挑样本。
3. 保存 155 个预期 trainable 参数的 requires_grad、optimizer membership、grad=None／零／非零／finite、梯度和 moment dtype；对 33 个 norm 保存完整小数组，对其他参数保存有界统计。核对实际 loss 分母和 padding。所有模型及 buffer 前后核验，optimizer/scheduler step 均为 0。
4. 在独立数组副本上计算固定梯度／矩状态下的理论更新及 BF16／FP32 舍入，仅作 shadow arithmetic；不调用真实策略 optimizer，不改变权重，不声称这是历史真实更新或修复效果。CPU 参考算术与实际 CUDA optimizer kernel 等价性另列未证。

判定分支：不在组／grad=None → 检查参数路由；在组但梯度真零 → 检查计算路径；非零梯度、理论更新被 BF16 舍入吞掉 → 确认精度限制候选；证据不匹配 → 保留未识别。任一分支都不自动授权 FP32 改造或重训。

预算草案：历史状态只读阶段 2 CPU、4 GiB、600 秒、无额外 swap；CUDA 工作上限 20 分钟、关机上限 25 分钟，RSS ≤8 GiB、allocated/reserved ≤8/10 GiB，证据 ≤1 GiB。当前机器若不足以满足，停止并重新评审；不通过 offload 或增加 swap 强行执行。20 分钟只是封顶，模型真实反向传播尚未实测。

### P3：根据 P1/P2 结果选择唯一训练轴

| 新证据 | 后续选择 | 不应做的跳跃 |
| --- | --- | --- |
| A/A 已分叉或输入身份不齐 | 先修已定位的复现／采集缺口，用新身份验证 | 不能把 A/B 差异当模型机制 |
| 确认 norm 更新路由或有效精度问题 | 保留历史端点；先 tiny controlled smoke，证明修复只影响指定参数，再登记单轴小实验 | 不能整体 FP32、改 LR、换 loss 一起重训 |
| 计算合同成立，但完整轨迹训练集仍拟合差 | 复用现有 full-trajectory probe 设计，补真实模型集成后才申请运行 | 不能把四轮／多更新试验称为只改变数据覆盖；它同时涉及训练预算／schedule |
| 训练轨迹拟合好、开发差 | 再测可观测阶段信息、场景映射和训练覆盖，各自独立设计 | 不能由简单近邻失败直接断言缺数据或必须加历史 |
| 离线拟合可用，闭环建立／保持失败 | 用 P1 的受控 trace 区分重规划时机、动作保持、物理跟踪与偏离分布；必要时另立恢复监督方案 | 不能拿偏离之后的时间索引专家动作当 recovery label |

完整轨迹 smoke/A/B 现有草稿记录为 2/500/2000 updates，均未获执行授权；具体采样、LR 序列和比较解释需重新审查。P2 的诊断结果也可能表明该训练探针才是下一个最有效实验，而不需要改变 dtype。

训练前明确验收、停止与资源界限；训练后按独立 reload → Gate 3 → 原五 seed Gate 4 评估。0.2 是既有最低任务成功率阈值，所有安全标准仍需同时通过；1/5 通过门槛也不代表广泛部署鲁棒性。训练集 loss、离线 MAE、探针通过均不能替代 M2 验收。

## 4. 本轮交付与边界

- 已完成：桌面资料分析、93 文件身份核验、原生 Gate 摘要算术与绑定核对、1,500 条参数记录对账、Basin 派生源核对，以及上述优先级和决策路径。
- 新增本报告与核验 JSON，架构入口仅追加最新证据导航；保留桌面原文件、已有工作树修改和全部历史报告。
- 未执行：真实模型 forward/backward、optimizer、rollout、远端访问、训练、提交或推送。平台当前状态不由旧关机记录推定。
- 唯一根因仍未确定，M2 仍未完成。下一次投入的目标是区分复现问题、有效更新问题和轨迹可学性问题。
