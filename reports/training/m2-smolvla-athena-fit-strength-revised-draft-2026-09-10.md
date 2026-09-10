# Athena-002：拟合强度修订草案

状态：`draft_runtime_prechecks_pending`，**不可启动训练**。本文件替代原草案中“保持 decay 256，延长至 1280”的候选；原草案和 coverage40 负结果保留。静态拒绝依据见 `reports/training/m2-smolvla-athena-static-audit-2026-09-10.md`。

## 假设、对照和唯一已选择的干预

问题：40 场景、256 步的 B 在训练夹爪组仅通过 1/4，开发视觉验收 0/4。候选 C 检验增加原生优化强度是否先补足训练拟合，再改善跨场景对应。

- 对照：保留 B step 256，model SHA `4e91624b34c3a16e22e0cbba19d8bcf3352a0d3835522ea6973ee22f0397accf`，不续跑其 optimizer，不重训 A/B。
- 候选 run：`m2-smolvla450m-visual-athena-fit40-002`；同一 base 全新初始化，seed 20260809、frame 0、既定 train40、batch 4、workers 0、grad accumulation 1。
- 干预明确包含 **steps 256→1280 和 cosine decay steps 256→1280**，warmup 16 不变。其余 optimizer 超参数、冻结范围、原生损失、processor、Action Contract、模型/数据 revision 均不变。不启用 paired loss、dropout、unfreeze、first-action weighting 或新数据。
- 5120 次成功更新样本暴露，预期每场景 128 次。这不是纯重复次数效应；更新次数与退火路径的效果不能拆开。前 256 步参数也不应被要求与 B 相同。
- 开发五场景已用于开发；不加载 hidden，不据其结果筛选新训练 seed 或更改阈值。

## 无卡预检模块，优先完成

最多 10 分钟的 CPU 合同检查，不加载模型或真实数据，不调用训练 launcher，不启动 GPU 作业；仅用单个合成 CPU 参数创建 optimizer 校验 LR。所有源码与输出仍在服务器，回传小型摘要。

1. 核验实例身份、现有环境和磁盘；确认与 B 的代码/环境关联，不因为 SSH 主机相同就认定数据盘相同。
2. 实际导入既有 schema，证明旧候选仍被拒绝、新候选结构通过；验证保存/日志网格、active episodes、fixed-frame 协议和每个执行 phase。输出测试证据，不能只依赖静态阅读。
3. 使用原生 scheduler 和单个合成 CPU 参数核对两个完整 LR 序列、末端状态；这是合成 optimizer 测试，不是模型训练。禁止读取 A/B optimizer state，禁止更新任何 checkpoint。
4. 使用真实 DataLoaderShard 和已注册 sampler，测试 2、10、256、320、1280 steps，分别核验 consumed 与 yielded 的完整顺序、epoch 边界、预取尾部、每场景次数。全是合成索引，不打开数据缓存。1280 steps 的尾部 expected 0 尚待实测。
5. 完成只读观察器的边界验证：preprocessor/optimizer 抛错时不计成功 update，少批、多批、错序、重复、错误 episode/frame 均拒绝；记录不得改 RNG 或 batch。先选择最小独立观察器，不侵入已 hash-bound 的 native trainer。
6. 查验 B 保存清单与 checkpoint 所需空间。四个候选 quarter checkpoint、一个 smoke checkpoint、原子保存临时副本及至少 2 GiB 余量全部计入；当前 free bytes 未核验，不能预设足够，不删除任何历史文件。

CPU 模块的退出标准是上述合同检查有真实可追溯证据。未测项、环境漂移、依赖缺失或不足磁盘均阻断 GPU 模块；不自动安装依赖或换环境。

## 受限训练阶段的入口必须显式核准

当前固定样本代码仅允许 `smoke` phase 下的 `bounded_visual_overfit`。候选保持这一受限 frame-0 诊断范围，不伪装成正式全轨迹训练。拟采用与 coverage40 相同的 v2 原生入口，但先单独完成 batch 1/4 forward、两步 smoke、冻结审计与独立 reload，再启动另一份明确登记的 1280 步诊断计划。两个阶段拥有不同 run name、计划 hash、输出和全新状态。

增加 optimizer_smoke.steps 本身不是启动授权，也不能绕过两步门禁；若真实 schema、runtime 或前置阶段的能力边界不允许该受限诊断，停止该候选。不得改成正式入口绕过 fixed-frame 拒绝，也不得放宽现有校验。原生 scheduler 在两步 smoke 中会自动缩放；必须分别核验 smoke 与 1280 步主诊断的实际 LR，不能把两者混为同一曲线。

## GPU 模块的候选登记

- 独立最多 30 分钟：身份/真实预检 6 分钟、训练约 10 分钟、B/C 统一评分和独立 reload 8 分钟、落盘与关机 6 分钟。时间只作基于 B 实测的估算，未启动新监督器，未挪用任何旧截止时间。
- allocated ≤8 GiB、reserved ≤10 GiB、RSS ≤10 GiB；batch/steps/退火和资源不足均不能自动降配或延时。
- 主诊断保存点拟为 320/640/960/1280，log_freq 16，save_freq 320；每个 quarter 保留完整 recovery state。只固定选择最终 step 1280，前三个不能按开发成绩择优。磁盘所需字节与实际保存行为须在 CPU/预检阶段确认。
- 统一评估须新增明确的 B/C 身份绑定。旧评分器锁定 A/B 256 步，不移除旧断言；为预算变化新建专用合同，同时复用原评分公式、固定噪声、完整预测与 reload 比较。
- 保留来源、base/processor/data manifest、Action Contract、逐步 LR/消费索引/成功更新数、冻结参数审计、Trackio、checkpoint、全部负结果和人工介入记录。

## 验收、停止与后续分流

沿用全非自身平均错配误差 C/M/F；train40 与 dev5 分别检查全维度、关节、夹爪，要求至少相同 3/4 噪声条件同时 C<M 且 C<F。与 B 比较开发误差/视觉收益，并要求完整 chunk 和首动作的关节/夹爪标准空间误差不回退。独立进程 reload 对全部七组完整数组逐值一致；错误、非法动作和数据漂移立即停止。

训练拟合仍失败，关闭本训练预算/退火候选；训练拟合通过但开发失败，转入单独登记的跨场景表示/对应关系诊断；离线全部通过，才准备完整轨迹和 Gate 3/4。任何分流都不自动追加新炉、解冻或改变 loss。M2 只有在闭环任务验收通过后才可能完成。

当前状态：官方源码与历史 worker hash 匹配、静态限制已核验；以上新的 CPU/GPU 执行、observer、磁盘、1280 步轨迹和 B/C 结果均 **not measured**。
