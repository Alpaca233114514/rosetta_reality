# SmolVLA frame-0 覆盖 8→40：审阅计划 — 2026-09-10

**文档模块已准备；实验未执行，尚不是可启动的 hash-bound 训练登记。**
状态：`draft_requires_remote_identity_and_authorization`；`executable=false`。
同名 JSON 记录固定设计与待核验项，不兼容 v2 launcher，不得当作训练配置。
本轮已完成公开资料查阅、本地静态核查、文档及另行授权的限时只读 SSH。
用户随后授权 Git 拉取与推送当前功能分支。一个小时是本轮模块工作上限；
旧 30 分钟预算已用完，新的开机、模型执行或计算预算仍未获授权。

## 1. 依据与当前证据

官方依据与项目推论的分界见
`reports/training/m2-smolvla-visual-coverage-official-evidence-2026-09-10.md`。

用户交接记录：原生 frozen-VLM + action-expert，8 个训练 episode 的 frame 0，
batch 4、256 步；训练场景视觉对照 4/4 通过，5 个开发验证场景 0/4。
工程回归 47 项通过；345 个 VLM 张量数值未变，expert/投影有更新，独立重载一致。
这些历史结果已由本地保留的远端结果摘要交叉核对；本轮仍未取得远端
原始报告、完整评估 JSON 和 checkpoint 哈希，未重新运行模型评估。
它支持“训练图像与动作对应可学”，视觉泛化仍未修复，M2 未完成。

| 历史结果（原 cyclic 协议） | 正确图 MSE | cyclic 错配 MSE | 视觉无关均值下限 |
|---|---:|---:|---:|
| 8 个训练场景 | 0.026303 | 0.375370 | 0.192625 |
| 5 个开发验证场景 | 0.217349 | 0.184712 | 0.121431 |

本地脚本及日志还保留了 episode 列表与逐噪声摘要；最终代码、指标有效维度、
配置和权重哈希仍待远端原始证据核验。
这些聚合数字不得用作新 all-pairs 协议的实测结果或新阈值。

权威工作副本：远端 durable root 下 `dev/visual-utilization-20260909-001`，
分支 `codex/smolvla-visual-conditioning-repair-20260909`。
证据：远端 `reports/training/m2-smolvla-native-visual-small-2026-09-09.{md,json}`
与 `runs/visual-native-small-001/`。
本地同名分支不是同步源码；本轮确认本地 `fixed_samples.py` 仍为单 episode
合同，缺少交接所述跨 episode 扩展。禁止用本地树覆盖远端副本。
关机发生于历史交接。新提供的 SSH 已核实属于另一台老实例，未找到本轮
权威工作副本；该只读会话已退出。原实例当前状态和文件完好性仍待核验。
本地恢复清单及 5 份历史源码片段见
`reports/training/m2-smolvla-native-visual-local-recovery-2026-09-10.md`；
片段尚未与远端最终哈希对齐，不能宣称恢复了通过 47 项回归的源码身份。

## 2. 假设、两臂与固定项

假设：在相同优化更新预算下，扩大 train frame-0 场景覆盖会改善正确图相对
错配图的开发集动作预测。干预同时改变每场景重复分配，不单独识别覆盖效应。

| 项目 | A：旧 8 场景控制 | B：40 场景候选 |
|---|---|---|
| 优化 | 只复用原 step-256 checkpoint，不重训、不续 optimizer | 同一 pinned base 全新初始化，主试验只运行一次 |
| 样本 | 原 8 个 train episodes，精确顺序待核验 | 既定 40 个 train episodes，各取 frame 0 |
| 主试验预算 | 历史 batch 4 × 256 步 = 1,024 暴露 | batch 4 × 256 步 = 1,024 暴露 |
| 平均重复 | 128 次/场景 | 25.6 次/场景；非整数不应伪称每场景完全相同 |
| 评估 | 在新输出根重新评估 | 相同 evaluator、样本、非视觉条件和噪声 |
| endpoint | 固定 step 256 | 固定 step 256；不得用开发集挑另一个 checkpoint |

必须逐字段与旧 A 的 resolved config 比对并保留：model/VLM/processor revision，
Action Contract、表示与 train-only normalization，原生 flow loss、时间分布、
50-step chunk、冻结/可训练参数集合、初始化/采样/噪声 RNG，batch、steps、
梯度累积、optimizer 全字段与参数组、scheduler 全字段及完整 LR 序列，
precision、compile 模式、相机优化、数据增强开关和已启用 feature 顺序。
不读取旧 optimizer 用于更新；smoke 也不作为 B 初始化，不启用 paired loss、
dropout、jitter、unfreeze、新 pooling、augmentation、EMA 或不同 LR。

40 个 train episodes 的本地登记顺序（须与远端身份逐项一致）：

```text
49, 4, 23, 43, 21, 37, 18, 34, 0, 47,
38, 29, 3, 26, 14, 17, 44, 30, 15, 42,
10, 35, 25, 32, 19, 36, 41, 28, 8, 27,
16, 11, 2, 20, 9, 39, 46, 48, 12, 40
```

开发验证固定为 `[22, 13, 7, 33, 45]` 的 frame 0；不增加开发场景、seed 或 offset。
hidden `[31, 6, 1, 24, 5]` 始终封存；读取 split 元数据不代表加载这些样本。
数据扫描须在 materialize/decode 前应用 allowlist，不能先加载 50 个场景再过滤。
保留每场景真实图像哈希和重复图像计数；episode 数不能冒充独立视觉变化数。
每个 episode 验证原始 row、解码帧、timestamp、相机、state/task、50 个动作
标签和 padding 对应；frame 0 必须具有完整合法 chunk，禁止跨 episode 拼接。
统计复用旧 train-only 产物，不因 8→40 重新拟合 normalization。

采样算法必须复用远端扩展及其 RNG 规则，并在优化前输出完整 1,024 项采样
顺序与每 episode 次数。若核实为整轮 permutation，则 40 场景应为 25 个
完整轮次加 24 样本，即 24 个 episode 各 26 次、16 个各 25 次；8 场景每个
128 次。若旧实现不是该算法，不按此假设擅自替换；先补充审阅登记。

## 3. 身份核验：本地候选值不充当远端证明

下列内容本轮从本地 YAML/源码静态读取、计算 SHA-256。与远端不一致时停止，
核对旧 A 真实合同；不得挑一个较方便的身份继续。

| 对象 | 本地候选身份 |
|---|---|
| LeRobot revision | `c903b114a90e703b3f7d0c46cb38727c328c55ff` |
| SmolVLA base revision | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| SmolVLM2 dependency revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| ALOHA data revision | `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` |
| base experiment YAML SHA | `7306014717eb3fe5a5997a92406ec18cb251f1c17cc9d46ed74d75ac71d8b19f` |
| bounded-gripper parent YAML SHA | `0e9dd0499d0708939ac73cc5d517849f133cf6deab072d9cde09f2880ae22210` |
| physical Action Contract YAML SHA | `fc71a0438f0e3af7258e5b52d82fa22fc53c12b47901606cbee715524392ac62` |
| AutoDL runtime profile YAML SHA | `be2bfc3ea2a518c85e56410ba3ea1da6f744236d51b6f4a5f6a7b73927e9f992` |

在取得原实例或其完整迁移副本的访问方式后，先取得**小型清单和摘要**，
大文件哈希在服务器计算。已完成的老实例检查不满足以下核验：

1. 当前平台/实例身份和计费状态；工作副本分支、HEAD、dirty diff、untracked
   源文件清单及逐文件 SHA。HEAD 不能代表未提交修改；先保留远端 before 快照。
2. 旧报告 md/json、旧 plan `m2-smolvla450m-visual-native-b4-pilot-003` 的
   实际路径与哈希、所有父配置、最终展开配置、原评估代码及 47 项测试的日志。
3. A step-256 权重、processor、normalization、optimizer/scheduler、RNG、
   manifest 和 reload 报告的精确路径/大小/哈希；权重仅以只读方式用于评估。
4. 模型/processor/tokenizer/数据缓存 revision 与逐文件 manifest、train 统计
   来源和 split；不下载缺失缓存，不沿用临时凭证，不输出 token 或环境全集。
5. 实际 sampler、trainer、launcher、Trackio 清理路径、loss/scheduler 源码
   SHA 与参数更新范围；确认 40 场景不会触发静默截断或另一个 phase 默认值。
6. 当前 GPU 名称/容量、Python/Torch/LeRobot/Trackio/decoder 身份，容器身份、
   durable mount、剩余空间、原 checkpoint 大小，以及 tmux/守护/退出流程。

核验后生成 create-only 的 remote identity、evaluation plan、preflight/smoke
plans，再以预检证据生成主试验 resolved plan；任何阶段都不改写旧计划。
新增内容与远端已保留源码合成服务器端 versioned workspace，优化前冻结整个
源码清单。只继承其已核实源码；不把本地 checkout 作为服务器替代包。
本轮没有生成运行用配置、实现脚本或模拟通过的 preflight。

## 4. 两臂共同评估协议（新登记，不改历史 cyclic）

### 样本、噪声与可复用预测

- 三个报告视图：原 `train8` 拟合视图；共同 `train40` 视图；共同 `dev5` 视图。
  A/B 都报告三者；A 在新增 32 个 train 场景上的表现不叫 A 训练拟合。
  图像 donor 仅在当前视图内部交换，dev 不混入 train。
- 新协议固定噪声：全零及 CPU generator seeds
  `[20260905, 20260906, 20260907]` 的标准 Gaussian。
  这是本计划指定的新评估值，不声称已核实旧 pilot seeds。
  每种条件只生成一份完整噪声张量，按原生输入 shape 扩展，A/B/所有场景
  复用相同值；记录生成器、shape、dtype、转换顺序和 tensor hash。
- 必须运行原生 **10-step 去噪**，模型 eval/inference mode，逐样本 batch 1，
  清空动作队列/历史/KV 等上下文；不能用含目标动作的 teacher-forced
  单步 velocity loss 代替。目标仅进入独立评分路径。
- 对所有场景验证 processor 后 state、instruction tokens/attention mask、
  相机有效 mask、placeholder、所有其他非视觉输入逐值相同；同一条件仅
  真实图像改变。不伪造相同 state；发现不相同则当前解析对照不成立，停止。
- 使用旧 online processor。真实相机不能误用 placeholder、错误 range 或
  全 false mask。按 manifest 对每次 ingress 检查；不新增视觉 embedding cache。
- 每个 arm/noise/image 只需预测一次完整 chunk；得出 `P[j]` 后与不同
  destination 的目标 `Y[i]` 评分。复用的是同一完整预测，不是重新采噪声、
  重跑 optimizer 或先平均 donor 预测。复用前以上相同条件须实测通过。

### 精确评分定义

主评分空间：**模型归一化动作空间，在逆归一化/有界夹爪解码前**；取合同的
有效 14 维、完整 50-step chunk，不计入 base policy padded action 维度。
需要远端核实获得该张量的代码位置并绑定 SHA，不能从经过周期性解码的动作
倒推 latent。本次定义与旧报告的可比性尚未核实，禁止直接比较其绝对数值。
评分运算转为 float64；所有目标/预测、共同有效 mask 和评估实现哈希保留。

对某 arm、noise、视图 E（大小 N）及维度组 g：

```text
d_g(p, y) = mean((p - y)^2 over valid horizon and dimensions in g)
C_g = (1/N) * sum_i d_g(P[i], Y[i])
M_g = (1/N) * sum_i [(1/(N-1)) * sum_{j != i} d_g(P[j], Y[i])]
mu_E = (1/N) * sum_i Y[i]                 # 每坐标的均值动作块
B_g = (1/N) * sum_i d_g(mu_E, Y[i])
visual_gain_g = M_g - C_g
below_mean_gain_g = B_g - C_g
```

`B_g` 是该有限集合上的最佳固定预测 MSE 下限；当非视觉条件/噪声固定，
忽略图像的确定性预测是固定动作块。开发标签仅用于计算这个解析 oracle，
不能传给模型、拟合 normalization、初始化或更新参数。另报 train40 目标
均值在 dev5 上的 MSE（实际 train-only 常数基线），与 `B_dev` 明确分列。
各视图自己计算 `B`；A/B 在同一视图共享同一个 `B`。

全体、关节组与夹爪组分别计算；维度从 Action Contract 名称/单位解析并验证
为 12 个 radian joints、2 个 normalized grippers（位置 6、13），不能静默硬编码。
另对解码后、最终 safety projection **之前**的标准动作分别报告关节 rad、
夹爪 [0,1] 的 chunk/first-action MAE、MSE、逐维误差、夹爪 latent 支持域
与 action legality；混合物理单位的总误差不作验收。

保存 N×N error matrix 和逐 episode 增益；N=8/40/5 时每种噪声分别为
56/1,560/20 个有向错配。它们有共享样本，不是独立观测。
epsilon 固定为 `1e-8 * max(1, abs(M_g), abs(B_g))`；
`M_g-C_g > epsilon` 且 `B_g-C_g > epsilon` 才计为正，边界/零增益不通过。
若某组 `B_g <= epsilon`，该组无法证明低于下限，标 `not measurable`，
不删除该组、改用 MAE 下限或宣告整项通过。

### 验收及判读分层

1. **完整性**：身份/采样/非视觉相等/资源/冻结范围/finite/合法动作全部通过；
   B action expert 必须有非零更新，全部冻结 VLM 数值保持不变。
   BF16→FP32 的无损保存转换单独报告，不记为学习更新。
2. **训练拟合**：A 的旧历史协议应能以记录精度重现（只用于完整性检查）；
   A 的新 train8、B 的新 train40 各自至少 3/4 条件满足主评分两条不等式。
   若 B 未拟合，结论为“固定预算下训练拟合不足”，不把它称为视觉信息缺失。
3. **开发视觉对照**：B 的 dev5 至少同一组 3/4 噪声条件中，全体、关节、
   夹爪分别同时满足正确图优于平均错配且低于解析均值下限。A 用完全相同
   规则评分；噪声条件是稳健性检查，不报告为四次独立重复或显著性检验。
4. **8→40 改善判断**：与重新评估的 A 比较，至少 3/4 条件满足
   `C_B < C_A - epsilon` 且 `visual_gain_B > visual_gain_A + epsilon`
   （全体归一化 chunk 主评分；此处 epsilon 为
   `1e-8 * max(1, abs(C_A), abs(C_B), abs(visual_gain_A), abs(visual_gain_B))`）。
   四条件平均的标准空间关节/夹爪 chunk 和 first-action MSE 均不得比 A
   增加超过 `1e-8 * max(1, abs(error_A), abs(error_B))`。
   单臂通过但臂间无改善仅报告“对照成立、覆盖收益未获支持”。
5. **独立 reload**：A/B 逐个独立进程从已保存权重与 processor 加载，重复
   相同 45 场景、四种噪声的完整预测；保存空间及标准动作均要求逐值相等。
   同一环境不接受只比聚合指标或只比 first action；差异保留，不能放宽后算通过。

以上为审阅阶段明确提出的分组/数值/臂间规则，尚非运行证据；最终登记必须
在新评估结果被读取前绑定这些规则。任意失败保留负结果并停止此设计。
即使全部通过，也仅为固定 frame-0 开发集进展；视觉泛化全面修复、完整轨迹、
任务成功、M2 验收均未被测量，不自动进入 Gate 3/4 或下一炉。

## 5. 预检和后续执行梯次（仅计划）

| 阶段 | 必须产出 | 当前状态 / 下一步条件 |
|---|---|---|
| L0 本轮 | 官方核对、方案 md/json、本地静态一致性 | 文档工作；不访问 ML 运行时 |
| R0 只读远端核验 | 第 3 节完整身份清单、实际入口和资源清单 | 老实例已查，权威副本与旧控制 checkpoint 仍缺；不加载模型/数据 |
| R1 冻结实现/登记 | server versioned workspace、evaluator/sampler 与 plan/config SHA、命令清单、独立截止时间、守护记录 | 缺失任何事实或预算则不可执行 |
| P0 当前环境/静态检查 | `python scripts/check_env.py`、实际 doctor/benchmark、原 47 项关联回归与 Ruff、评估算术/身份拒绝检查 | 新预算内在 AutoDL 平台容器执行；不嵌套 Docker，不安装依赖 |
| P1 真实数据/forward 预检 | offline cache inspect、相关 `pytest -m data`、Gate 1/2 的匹配证据、40/5 frame-0 identities、dummy 与 batch-1 CPU/小样本路径、CUDA no-optimizer forward | 单样本起步，再 batch 4；记录 optimizer_created=false |
| P2 独立 smoke | fresh-base batch 4 两步，finite/梯度/参数范围/资源、完整 checkpoint 与 independent reload | 输出独立，不续入 B，不算进 256 主试验步数 |
| P3 控制验证 | A 旧指标重现、新 train8 拟合、A/B 共用的新评估协议/代码冻结 | A 完整性失败即停止，禁止临时重训 A |
| E1 一次 B | fresh-base batch 4 × 256；resolved config、逐步 loss/gradient/LR/采样摘要、最终完整 checkpoint | 仅后续明确授权；smoke 后重新从 base/原 seed 创建全部状态 |
| E2 共同评分/独立 reload | A/B train8/train40/dev5、新 error matrices/预测与逐组结果、最终判读 | 固定 step 256；不得挑最好噪声或删失败场景 |
| C 收线 | append-only 结果、完整远端保存、脱敏小摘要和哈希清单 | 不自动下一炉；电源操作只按当次明确授权 |

本地 `scripts/run_autodl.sh smoke/formal` 实际转发到历史 Way runner，不能凭
名称直接调用。远端核验 v2 路径 `scripts/run_smolvla_v2.py`、实际 pilot
入口及 sampler 如何绑定 phase；命令逐项写入新 manifest 后才运行。
禁止把 `smoke` 参数任意改成 256 步绕过原两步门禁，也不假定改用 `train`
仍会取 frame 0。必须从展开配置与首批真实样本证明原 pilot 的路径被保留。
若需工程修补，在服务器单独保留补丁/回归证据，学习合同有变化则退回审阅。
Gate 1/2 只有在证据身份与当前合同/adapter 匹配时才可复用；不匹配或缺失
则暂停，另行登记及授权所需验证，不在此计划下自动启动仿真补测。

## 6. 资源、流量和停止条件

以下是**建议的未来预算，不是当前授权，也不是已测能力**：

- R0 只读核验拟限 10 分钟；新计算链 P0–C 拟设独立 30 分钟总截止，包含
  preflight/smoke、A 复评、B 训练、共同评估、reload 和保存。用户可在授权
  时给出不同上限；必须先登记，不能复用上一轮 30 分钟或本轮一小时。
- 计算链分配参考：环境/数据/预检/smoke 10 分钟；控制验证 3 分钟；B 训练
  5 分钟；共同评估/reload 8 分钟；落盘与小摘要 4 分钟。每阶段开始前检查
  剩余总时间是否足够；预检用尽预算便停止，不挤掉 reload，也不自动延时。
- GPU 建议上限：allocated 8 GiB、reserved 10 GiB，且服从已登记运行时
  更紧的限制；远端 CPU RSS 建议上限 16 GiB。均需当前 preflight 验证。
  历史 pilot 约 132 秒、peak allocation 2.82 GB 仅是交接估计依据，不是预测。
- 新评分基础成本：每臂 45 场景 × 4 噪声 = 180 次完整 chunk 预测；两臂
  首次评估加独立 reload 共 720 次、每次 10 个去噪步，即 7,200 个 expert
  去噪步。旧协议重现、smoke 和预检另计；不把 6,544 个有向错配评分/臂
  误当成同样数量的模型调用。完整预测复用，不改变 frozen features 缓存策略。
- 磁盘不预填虚假空闲值。按实际 checkpoint 格式核算所有新保存点、smoke、
  export/原子保存临时副本、源码/日志，再加 2 GiB 余量；创建前核对 durable
  mount 与空闲空间。预算不足就停，不删除旧 checkpoint 或迁走历史证据。
- 所有模型/数据/预测/完整日志/checkpoint 保留服务器。仅传新小文档、审查过
  的指标摘要和哈希清单；不传样本或大文件，不新增下载。Trackio 新 run name
  与 durable store 预先绑定，仅本地 durable 记录；公开 Space 同步待另授权。
- 长进程用 tmux 或已核实等价守护；启动/首批/退出边界核查，中途稳定阶段
  按 Bash `sleep 300` 最小采样。独立 deadline watchdog 负责到时停止本次
  run，不能高频远端轮询或杀其他任务；预算内优先预留落盘时间。

立即停止：任何身份漂移、缓存缺失、hidden 访问、样本/非视觉条件不符、
未授权第二实验轴、NaN/Inf、OOM/超资源、unsupported accelerator、非法动作、
冻结参数更新、采样次数/optimizer/scheduler 偏离、checkpoint/Trackio durability
或 reload 失败，以及预算不足/到期。原错误和 partial evidence 保留；未运行
项写 `not measured`，不写 0 分。不得自动降 batch、延长步数、改阈值或重启。
验收负结果关闭的是本次固定预算设计，不证明所有视觉泛化方案无效。

## 7. 交付身份与完成边界

拟 run name：`visual-native-coverage40-001`；拟证据根：
`runs/visual-native-coverage40-001/`。实际远端若已存在，则登记一个新的唯一
编号并记录原因，禁止覆盖；本轮没有访问或创建该远端目录。

未来证据至少包括 `identity/`（before/after/config/cache/environment）、
`plans/`（分阶段 hash-bound 登记与实际命令）、`preflight/`、`smoke/`、
`training/`、`evaluation/`（A/B predictions、targets、masks、noise hashes、
error matrices、group metrics）、`reload/`、`tracking/`、`closure.json`。
当前 remote checkpoint/optimizer/scheduler/seed/normalization hashes、完整
命令和最终资源许可全部为 pending；同名 JSON 里的 null 表示未知，不能默认。

文档验收只检查 JSON 可解析、固定设计一致、split 不重叠、预算/评分数量
算术、相对路径与引用、本轮新增 diff 和已有修改保留。后续增加了授权的
只读 SSH、Git 身份核验与本地证据清点；没有运行模型测试、训练或仿真。
47 项回归是历史日志记录，不能写成本轮重跑通过。

## 8. 评估器实现前的算术验收样例

以下是从第 4 节公式直接算出的单维、单时间点预期值，不是模型/数据结果。
供后续远端评估器单测使用，本轮没有新增或运行 ML 单测。

| 目标 Y | 图像对应预测 P | C | M（所有非自身） | B | 应有判读 |
|---|---|---:|---:|---:|---|
| [-1, 1] | [-1, 1] | 0 | 4 | 1 | 正确视觉对照为正 |
| [-1, 1] | [0, 0] | 1 | 1 | 1 | 忽略图像，不通过 |
| [-1, 1] | [1, -1] | 4 | 0 | 1 | 反向对应，不通过 |
| [0, 2, 4] | [0, 2, 4] | 0 | 8 | 8/3 | 若先平均错配预测会错误得到 M=6 |
| [1, 1] | [1, 1] | 0 | 0 | 0 | 下限退化，不能证明视觉收益 |

任意固定预测 q 都满足 `mean_i (q-Y[i])² = B + (q-mu_E)²`，因此不可能
严格低于 B；忽略图像时 P 全相同，还必有 C=M。验证这些反例，另覆盖
对角项排除、集合大小 <2、NaN/Inf、mask/非视觉条件不一致、padding 维度、
两个单位组方向相反、错误 donor split 和缺失逐场景证据时拒绝评分。
