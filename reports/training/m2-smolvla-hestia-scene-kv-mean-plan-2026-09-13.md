# Hestia 同 checkpoint 图像 K/V 场景均值干预

科学身份 `hestia-scene-kv-mean-20260913-001`。
前置证据是今早早期差距分解，以及本工作区完成的独立 85,632 项标量复核。
本文件冻结科学设计；没有发放可运行 permit，GPU 状态为 `not measured`。
执行实现、源码/config/calibration 及传输包未封存前禁止启动计算实例。

## 假设与唯一干预轴

在 640 步，真实图像位置的跨场景 K 或 V 变化可能产生过强或错误的动作场景响应。
唯一干预轴为移除的投影输出位置：不移除、图像 K、图像 V。
640 为预先指定的主要诊断层次；1280 为后期对照层次，分别报告，不择优。
不交换 checkpoint 权重，不复用昨日 640→1280 donor 差缓存，不改训练超参数。

- backbone：revision-pinned SmolVLA 450M，冻结 VLM。
- adaptation / Action Expert：保留 Hestia 原生已训练 expert/state/action projections。
- dataset：原登记 ALOHA insertion train40/dev5 的全部 45 个 frame0 场景。
- noise：原始四个固定推理噪声，逐噪声执行与配对评分；每次 10 denoise steps。
- checkpoint：640 与 1280 的已核验原生文件，维持既有 action、processor、normalization
  和 dataset revision。具体 SHA 从昨日封存模板与本次本地核验绑定，不能取当前旧分支默认值。

## 干预定义

对每个 checkpoint 分别从原生 forward 采集奇数层 1/3/5/7/9/11/13/15 的 K/V 输出。
保留完整 241 位置的原生 CUDA BF16 GEMM，仅真实图像位置 `[0,64)` 可以替换。
每层均值仅对 train40 的场景轴取均值，保留 64 个 token 的位置和通道：

`mean[s, layer, projection, token, channel] = mean_train40(native_output)`。

不读取 train/dev 动作标签来构造均值；不跨 checkpoint 或噪声混合校准。
必须先证明这些前缀输出跨四噪声和十个 denoise 调用逐值一致，才使用共享均值；
若不一致则本假设的实现前提失败，不能临时改为噪声相关缓存。
均值采用 float64 累加，再一次转换为 BF16，直接赋值；记录转换方法与缓存 SHA。

K-only 条件仅替换真实图像 K，保留 V；V-only 条件仅替换真实图像 V，保留 K。
其他 177 个 prefix 位置，包括空相机、语言和状态，逐值保持原输出。
不裁剪或重标定注意力；保留现有 mask、位置编码、query 和完整后续原生运算。
**不声称 V-only 条件下整个去噪过程的注意力固定**，后续 query 可以随干预改变。

## 条件顺序与可逆对照

总计十条件，每条件 45 场景 × 4 噪声 = 180 forwards，共 1,800 forwards，0 optimizer：

1. `base640_native`：逐值复现已有 640 数组，采集并封存本 checkpoint 校准。
2. `base1280_native`：逐值复现已有 1280 数组，封存独立校准。
3. `base640_krestore`、`base640_vrestore`。
4. `base1280_krestore`、`base1280_vrestore`。
5. `base640_kmean`、`base640_vmean`。
6. `base1280_kmean`、`base1280_vmean`。

restore 使用与 mean 相同的 hook / 替换分支，但在送入 attention 前将所选位置直接
恢复为封存的该场景原生输出；不能通过 BF16 加减残差假定可逆。
两原生端点和四 restore 的完整 normalized/standard 输出必须逐值复现，任一失败
立即停止，禁止执行 mean 条件。校准含 checkpoint 文件 SHA、模块/位置、train40
episode 顺序、mask、每场景输入和原生输出 hash，且绑定已通过的原生结果。

这些 restore 是“去掉场景分量后恢复原值”的实现阳性对照，**不是独立的跨 checkpoint
互逆参数因果实验**。不能借 restore 逐值通过宣称获得新的双向科学证据。
干预是同 checkpoint 的成对移除/恢复；科学效果由 mean 与原生条件的配对差衡量。

## 指标与解释边界

主要读数为 640 dev5 完整左关节的配对 MAE/MSE 与场景协方差变化。
完整报告两个 checkpoint、train40/dev5、四噪声、standard/normalized、全段/首动作、
左右及合并关节/夹爪，并保留逐场景误差。原生指标须复现独立复核结果。
保留训练标签均值与中位数常数，及已封存的 train40 输出模板；不新增调参网格。

- 四噪声都改善 MAE/MSE且降低 MSE 场景不匹配项：支持被移除位置承载有害场景响应。
- 只有部分指标或噪声改善：报告混合结果，不扩大为该路径的唯一因果贡献。
- 无改善或变差：记录阴性结果，不能据此排除该路径，因为均值替换可能产生分布外激活。
- 即使改善也不是新 policy、任务成功或 M2 完成；其他上下文位置仍可能携带图像信息。
- 所有开发场景保留，不能用输出模板效果大小反推、选择或裁剪校准统计量。

本协议评估预先登记的一次场景依赖消融。存在 K/V 非线性和上下文耦合，效果不可加和；
不把输出 MSE 恒等式直接当内部 K/V 的因果分解。

## 执行门禁、预算与停止条件

先逐项核对来源实现 SHA，再创建本功能分支的新实现；旧 identity、脚本和失败证据不改。
先做 shape/mask/训练行隔离/校准篡改/恢复精确性/跨 checkpoint 拒绝的合成测试，
完成来源依赖核验和命令行集成检查，才封存可执行模板与传输清单。
本地测试从 WSL Bash 进入固定离线容器（2 CPU/2 GiB），不加载模型。

CUDA 使用既有已授权的单卡 RTX 4090 D，原生 torch `2.8.0+cu128`、NumPy `2.2.6`、
LeRobot `0.6.2`；不得以本地 XPU 结果代替 CUDA 一致性。
按既有 Chrome 流程精确匹配目标实例和 SSH 端口后开机，仅在可执行包封存且确有需要时操作。
不得为取回昨日 shutdown-request 单独开机；下次已有窗口先只读补回该记录。

工作预算 1,800 秒，受保护关机截止 2,100 秒；单条件 180 秒，CUDA allocated ≤4 GiB，
RSS ≤5 GiB，回传 ≤64 MiB。看门狗必须先就绪，接收器先于 worker 启动。
完成或失败都封存退出状态、白名单 manifest、回传并本地逐文件 SHA 复核、发送匹配收据、
按已授权保护流程关机并查看平台状态。不释放实例，不将 SSH 断连当关机证据。

身份、输出布局、mask、参数不变、校准隔离、原生复现、finite/动作合法性、预算或
看门狗任一失败立即停止并保留失败证据，不放宽容差、不自动重跑旧身份、不转训练。
本地/远端源码、配置、校准、输出清单和运行保护的封存门禁均未通过前，本计划保持不可启动。
