# Canonical traced Gate 005：失败阶段与证据边界

**0/5 已定位到抓取建立与保持失败：1000/1001 连物体都未接触；1003/1004 只操作了 socket，右侧 peg 未被接触；1002 仅在三个控制步短暂达到 reward 2，随后丢失。没有任何 episode 完成插入。** 这比“模型输出合法但失败”更具体，但尚未确定学习侧唯一根因。M2 仍未完成。

本报告只读已回传 JSON 与源码，并用 PowerShell 独立重算；没有模型加载、新 forward、optimizer、仿真步、SSH 或数据集读取。未修改历史证据、动作、解码器和验收阈值。

## 证据与计数约定

证据根为 `runs/canonical-posttrain-received-20260916-005-ssh/verified/`，主结果见同目录对应 `results/.../gates/gate4-smolvla-sim-471.json`。本轮分析重新核验五个 trace manifest 的 15 个文件 SHA，逐条读取 **7,500 个事件、2,500 个已完成控制步**。

下文 step 为零基索引；step s 的 post-step 快照对应名义仿真时间 `(s+1)/50` 秒，不是墙钟时间。开度 ≥0.5 与开过后 ≤0.2 仅用于描述时间线，不能当作任务验收或实际抓住物体的判据。接触计数是每个 post-step 的 contact point 计数；持续接触和同一 geom pair 的多个接触点可能重复计数，并非独立事故数。限位计数按每步每个越界 joint 统计。

`trace.jsonl` 的字段入口：

- `event=prediction`：`internal_grippers[slot][left/right]`、`support_limit`、`decoded_first_action`、`postprocessed_first_action`、`input_identity`。
- `event=step_started/step`：`step`、`state_before/after`、`executed_action`、`observed_grippers_before/after`。
- `physics_before/after.value.bodies.{peg,socket,vx300s_left/gripper_link,vx300s_right/gripper_link}.{position,quaternion}`：实际 MuJoCo body 位姿。
- `physics_after.value.contacts`、`joint_limit_violations` 和 `reward/success/terminated/truncated`：接触、物理限位、任务结果。

这里的 gripper_link 是实际 palm/body 原点，不是指尖、抓取中心或命令 FK。初始物体 body z=0.05 m 随重力落到桌面，不能把这段垂直位移当作策略操作。

## 五个 episode 的失败时间线

| Seed | 已观察的任务阶段 | 首次物体接触 / 关键事件 | 最终结果 |
|---|---|---|---|
| 1000 | 接近后停留，未建立抓取 | 全 500 步无机器人-peg/socket 接触；双爪命令与实测开度都从未达到 0.5；两物体 XY 从 reset 至结束逐值不变 | reward 一直 0，无碰撞/限位 |
| 1001 | 接近后停留，未建立抓取 | 左爪仅 step 25 命令达到 0.5，实测从未达到；右爪也未达到。全程无物体接触，两物体 XY 不变 | reward 一直 0；step 17 一次右手指下限越界 |
| 1002 | socket 抓取/搬动；右 peg 多次接触但未稳定保持 | 左/右实测开度分别 step 33/38 达到 0.5；step 73 首次左指碰桌，74 首次左指接触 socket；77 左命令降至 0.178；92 首次右指接触 peg；430–432 reward=2，433 回到 0 | socket 最大 XY 位移 89.49 mm、最高 body z=134.51 mm；peg 最大 XY 位移 89.61 mm；没有插入 |
| 1003 | socket 被搬动后失去持续接触，右 peg 从未抓取 | step 61 首次左指接触 socket 并出现手指限位；69 首次碰桌。socket 最高 body z=68.71 mm。step 212 后持续双侧 socket 接触消失，之后仅 417 一次单指接触；右 peg 全程无机器人接触且 XY 不变 | reward 一直 0；socket 最终 XY 偏移约 103 mm，落回低位 |
| 1004 | socket 短暂抓取并抬升，随后丢失；右 peg 未抓取 | step 92 双侧左指碰桌，97 双指接触 socket，101 socket z 约 72 mm；最高 z=125.59 mm。125 最后左手右指/socket 接触，126 左开度命令升至 0.603，127 最后单指接触，128 后无接触；140 z 约 27 mm | reward 一直 0；peg 全程无机器人接触、XY 不变；socket 最终 z=22.09 mm |

1000/1001 不是“已经拿住后插不进去”：到最后两个 palm body 的 z 仍约 0.21–0.26 m，物体留在桌面，缺失最基本的物体接触。1003/1004 同样还没有进入可靠双物体对齐/插入阶段。

1002 的 reward 2 只维持 **3 步，名义 60 ms**。step 430/431/432 右侧 `10_right_gripper_finger` 接触 peg；433 这一接触消失，reward 同步回到 0。peg body 在 432/433 从 z 约 26 mm 上升到 43 mm，并不能据此认定仍稳定拿住：之后 437 已回到约 11 mm。socket 到末步仍有手指接触、z=79.69 mm，但 peg 在桌面低位，双物体稳定操作未形成。这里根据实际接触和位姿描述阶段，不把单次接触等同于力闭合抓取。

所有 episode 的第一步都有限、无异常场景接触/限位，reset 实测两夹爪同为 0.09984833；第一条左右夹爪命令分别为：1000 `0.17537/0.02423`，1001 `0.20803/0.00422`，1002 `0.20607/0.00094`，1003 `0.25328/0.00788`，1004 `0.21263/0.00871`。这说明动作从一开始就有场景/噪声差异，**不能在没有同 reset 专家动作的情况下把 step 0 宣判为首次错误**。

## 限位与碰撞具体是什么

**27 个物理限位计数全部来自手指关节，12 个机械臂关节没有记录到越界。** 不能将“joint-limit violations=27”解释成机械臂 joint mapping 错乱。

| Seed | Joint | 零基 step 区间 | 计数 | 最大越界量 |
|---|---|---|---:|---:|
| 1001 | `vx300s_right/left_finger` | 17 | 1 | 下限外 0.178 mm |
| 1002 | `vx300s_right/left_finger` | 81–86 | 6 | 上限外 5.201 mm |
| 1002 | `vx300s_left/left_finger` | 199–200 | 2 | 上限外 0.290 mm |
| 1003 | `vx300s_left/left_finger` | 61–78 | 18 | 上限外 2.634 mm |

手指物理范围为 `[0.021,0.057]` m，检测容差 `1e-5`。例如 1002 step 81 的右命令开度仅 0.700，但实测归一化开度 1.105、物理手指 qpos=0.062143 m，且同一步右指碰桌。合法命令与接触中的实际手指状态不同；现有观察不足以把机械接触、动态超调或控制器各自贡献分离。

**60 个 unexpected contact 全是手指-桌面接触**：1002 为 38（左指桌面 6、右手两指桌面 5+27），1003 为 15（左手右指），1004 为 7（左手两指 4+3）。没有错误物体或机械臂互撞计数。完整 contact-pair 区间写在 JSON 的 `episodes[].contacts`，原 adapter 对正确 arm-object 两个手指都豁免；这批 60 不是历史单侧 grasp allowlist 的误报。

1000 没有任何碰撞/限位仍失败，1001 只有早期一次很小的手指下限越界且全程未接触物体。因此碰撞和限位是真实附加失败，无法独自解释所有五局 0/5。

## internal gripper support 与 mapping 检查

独立逐值核对全部 **35,000 个已执行动作标量**：`decoded_first_action == postprocessed_first_action == executed_action`，零差异；逐步 `state_before` 与前一步 `state_after` 的 34,930 个标量也完全相同。保存的内部夹爪值用 `(sin(x)+1)/2` 重算后，与实际命令最大差异 `5.57e-8`，符合 float32 输出的舍入量级。源码确认执行的是每次预测的 slot 0，左右夹爪在 14-D 顺序中的 index 6/13，观测 state 维度来源为实际 dataset features。

这些证据排除本轮 trace 范围内的静默额外 clipping、raw/projected/executed 错接或相邻记录断链；不等于已经证明模型学会了正确的图像/state/action 对应关系，也不独立证明所有物理单位语义。

| Seed | 左 executed latent 越界/500 | 右 executed latent 越界/500 | 保存值按 support 截断后解码的最大命令差 |
|---|---:|---:|---:|
| 1000 | 0 | 2 | 0.000765 |
| 1001 | 0 | 9 | 0.005406 |
| 1002 | 7 | 9 | 0.007629 |
| 1003 | 7 | 1 | 0.091173 |
| 1004 | 2 | 7 | 0.009992 |

合计 **44/5,000=0.88%** 已执行夹爪 latent 超出 `±pi/2`。最大折返出现在 1003 step 62 左侧：internal=2.18427，原解码=0.908827，而 support 截断的代数值=1。该局的 support 越界与接触/限位窗口部分重叠，值得保留为局部假设；但另外两局完全不抓物体时折返幅度很小，且左侧没有越界。它不是已证实的共同根因。上表只是已保存数值的代数比较，**没有执行 clipping 替代策略，也不意味着建议更大开度或放宽物理限位**。50-slot full-chunk 的其他越界值没有执行，不能计入造成当前物理动作的直接证据。

## 为什么 004 与 005 轨迹不同仍未定

两次 registration 共有 **1,082 个源文件 hash 完全相同**，已有文件仅 architecture 文档与 `scripts/audit_smolvla_inventory.py` 两处 hash 改变；005 新增 repeat wrapper、trace、trace verifier 和对应测试四文件。已额外核对当前审阅的 Gate engine、canonical runtime、native loader、processor、Gym adapter 与 Action Contract 都等于两次注册 hash。

两份实际 `gate.yaml` 从 `inference:` 至尾部的文本完全一致，包含 camera、instruction、BF16、receding slot 0、环境与噪声 seeds、500 步预算和阈值。固定 step5000 权重 SHA 相同，saved processors 固定，doctor/Gate runtime 所报 torch 2.8.0+cu128、LeRobot 0.6.2、Gym-ALOHA 0.1.4、RTX4090D 与 EGL 相同；独立 reload/native 数组也都通过原检查。

源码中每局 policy Gaussian noise 使用独立 CPU generator 并按对应 seed 重置；trace 不新增随机数调用，原预测返回值原样送回，physics snapshot 是读操作。trace 确实增加 CPU tensor copy、图像 hash、JSON 序列化与等待开销。以上静态审阅未发现明确改变动作或 seed 的路径，但不能代替实际 CUDA 配对实验，也不能证明驱动、渲染、物理二进制和所有 backend deterministic flags 完全相同。

004 没有逐步输入/动作 trace，因此目前无法定位两次首次分叉的 RGB、state、noise 或 action。**轨迹差异仍未归因：不能声称 trace 改善了策略，也不能直接认定 trace 有副作用，或直接把原因定为 CUDA 非确定性。** 同一固定离线输入的精确 reload 也不能代替闭环逐步等价。

## 后续证据应如何收窄

已定位的是闭环的“接近不抓取 / 只操作 socket / peg 抓取短暂且失去”阶段。学习侧可能涉及场景动作对应、开合时序、当前状态下恢复策略等；本报告没有足够对照将其中任何一项提升为唯一根因，不能据此直接重训或重新选 checkpoint。

下一次需另行登记并授权的最小对照，是同 reset 的 traced/untraced 成对 RGB/state/noise/first-action 检查，再按首次分叉扩展短轨迹；它先回答 004/005 可复现性。之后才对已定位抓取阶段做单轴控制实验。原五个 Gate seed 没有匹配的状态条件专家，所以“首次偏离专家”仍未测量；不能用旧 seed10 或时间索引 expert 动作补作本轮答案。

可复查机器摘要为同名 JSON，含五条 trace SHA、每个 contact/joint 区间、所有 support 越界执行步、第一步动作/state、选定物理快照、源码 hash 对照和重算验证计数。重算入口保留在 `runs/traced-gate-analysis-20260916*.ps1`，输出均 create-only；未运行新的模型/仿真测试。
