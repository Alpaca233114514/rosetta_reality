# 本地视觉到动作泛化诊断：早层 / 末层 K/V 单变量实验

预登记日期：2026-09-11。分支：`codex/smolvla-local-generalization-20260911`。
用户授权本地原因调查与一个变量的实验。本实验是冻结模型的读出诊断，
不启动 Hestia 正式训练或远端计算。机器合同由同名 JSON 固定。

## 已有证据与尚未识别的原因

依据 `m2-smolvla-native-visual-coverage40-result-2026-09-10.{md,json}`：

- 8 场景 A 与 40 场景 B 均在四个噪声条件下未通过 dev5 视觉验收。
  B 的正确图 normalized MSE 从 A 的 0.217349 降至 0.184360，
  仍高于 train40 均值在开发集上的常数基线 0.139549。
- B 开发关节预测的场景相关系数约 0.01955；目标方差 0.11387，
  预测方差 0.02228，均值偏差平方 0.05013。变化有了，方向却没有对齐。
- B 的 train40 夹爪仅 1/4 通过，预测场景方差 0.00401，目标 0.10712。
  不能把 aggregate 拟合通过解释成夹爪也拟合充分。
- 固定 1024 次暴露使 8→40 覆盖同时把每场景重复从 128 次降至 25/26 次。
  覆盖与重复被共同改变；尚不能隔离哪一个贡献了失败。
- A/B 完整数组独立 reload、processor、采样顺序、345 个 VLM 参数不变
  等证据通过，支持该结果是有效负结果；不支持“模型没读到图”。
- 旧 mean / 2×2 tower/connector ridge 均未超过常数基线。
  `m2-smolvla-vision-diagnostic-repair-2026-09-05.md` 明确：这些不是
  动作专家实际使用的上下文化 K/V，失败不能证明信息不存在。

帧 0 的机器人状态相同，指令相同；在固定噪声下，场景间差异来自图像。
本轮优先填补 K/V 读出证据缺口。训练量不足、示范首动作歧义、空间池化
损失、非线性映射不足仍是候选解释，不提前确定唯一根因。

## 唯一实验轴

比较同一冻结模型中，动作专家第 1 层与第 15 层（从零计数，分别为最早和
最晚 cross-attention 层）实际 `k_proj` / `v_proj` 输入中的真实相机 token。
它们是 VLM 层产生、经过原生 cache 传递、尚未进入专家 K/V 投影的表示；
不是 connector，也不是专家输出。K 已包含原生 RoPE，V 保留原生值。

通过只观察输入的 hook 在同一次原生推理中收集两臂，逐去噪步核对 K/V
一致；首样本另作无 hook / 有 hook 的完整 50 步动作逐值比较。
不修改模型 forward、权重、注意力或 processor。

固定条件：

- 本地既有 Zen-uniform step316 artifact（manifest SHA 见 JSON）。
  coverage40 B 权重尚未回传；本轮不冒充对 B 的直接模型重测。
- 原固定基座、VLM dependency、LeRobot、数据 revision 和 Action Contract。
- 40 train / 5 已参与开发的 development episodes；只读 frame 0。
  hidden `[31,6,1,24,5]` 在 Arrow 扫描期排除。
- 真实相机在 prefix 最前；无图像 special token；mask `[true,false,false]`；
  8×8 token 网格。只对前 64 个真实视觉 token 作同样的 2×2 空间池化，
  连接 K/V；两臂特征宽度必须相同。语言、state、空相机和 padding 不进读出。
- 使用原始标准动作空间的首动作标签。关节 radian 与夹爪 normalized 分组报告。
- train 内五折、固定 seed 20260911。仅用早层 train 折在
  `[0.001,0.1,1,10,100]` 中选择一个 ridge alpha，两臂共用它。
  各自的标准化、中心化、回归系数只由 train 拟合。
  alpha 选择沿用既有混合单位 MAE 代理；正式判读必须查看物理单位分组。
- 两臂同时报告全部结果，不用 development 选择层、alpha、pooling 或噪声。
  OOF 分数参与 alpha 选择，明确标为调参分数，不视为独立测试。

这回答“固定读出能否从更深层提取动作相关信号”。层间差异不等同于
因果修复：不同层可以用不同坐标系编码同一信息，线性/2×2 失败不是
无信息证明。首动作标签还可能包含示范者偏好，读出不等于可执行控制器。

## 运行与停止

`scripts/diagnose_contextual_kv_depth.py` 为 create-only 入口，
`src/rosetta_reality/vla/contextual_kv_probe.py` 负责无模型的读出与分解。
先完成本地容器 `check_env`、Ruff、相关 synthetic tests、dummy hook forward、
真实缓存 `pytest -m data`，再由入口核验全部 45 个 image/state/action/task
身份后加载模型。源码和 upstream SHA 在模型前检查。

本地 WSL Bash → 已有 digest-pinned Linux Docker / Intel XPU；网络关闭；
batch 1，2 CPU，容器内存与 memory+swap 均为 6 GiB，XPU allocated 上限
4 GiB。最多 46 次完整原生 forward（45 样本 + 首样本无 hook 对照），
模型诊断总预算 600 秒；无 optimizer、模型训练或 simulation。
输入/身份漂移、OOM、非 finite、hook 改变结果、KV 随去噪变化、超时立即停止，
保留失败记录，不自动改参数或回退设备。

创建新运行目录 `runs/contextual-kv-depth-20260911-001/`，保留输入清单、
两臂 feature/target/输出数组、SHA、完整指标、alpha/folds、hook 次数、
VLM 参数前后摘要及资源。数组落盘后 exact reload；这不是新的独立模型
进程 reload，也不改变原 artifact 的历史 reload 结论。原实验数据不覆盖。

## 预先固定的判读

末层读出支持信号必须同时满足：dev5 的关节和夹爪 MAE 都低于早层，
都低于 train mean 与 train median 两个常数基线，且正确图 MAE 都低于
所有非自身图各自 MAE 的平均值；epsilon 固定 1e-8。不平均错配预测。

- 若末层通过：支持下游 K/V 在这个读出和切片下含有可利用信号，后续
  应区分动作专家拟合与小样本泛化，不能宣称策略已改进。
- 若早层超过常数而末层退化：支持调查深层处理对当前线性读出可用性的影响，
  不能直接证明信息被丢失或据此删除模型层。
- 若两臂都失败：保留负结果；线性/池化能力、数据量和标签歧义仍未排除。

同时报告 train 拟合、train OOF、逐 episode/维度误差、场景方差、协方差与
均值偏差。五个 development episodes 不是新测试；本次没有独立训练重复。
无论读出结果如何，policy improvement / Gate 3/4 / task success 均为
`not measured`，M2 未完成。
