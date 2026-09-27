# Base 与 canonical 2500/5000：全部参数比较结果

在用户要求的新功能分支 `codex/prepost-parameter-comparison-20260923` 完成，
保留原有未提交修改。复用本地三个权重文件，没有重训、提交、推送或 SSH。
此前 Docker 暂停已解除；本结果补充而不改写前置检查报告。

## 实测结果

每个端点包含 **500 个同名、同 shape 的浮点张量，450,046,176 个值**。
三组均逐张量比较，没有抽样、截断或省略零变化参数。全部值有限。

| 比较 | 数值变化张量 | 数值完全相同 | 字节完全相同 | dtype 变化 |
| --- | ---: | ---: | ---: | ---: |
| base → 2500 | 122 | 378 | 179 | 199 |
| base → 5000 | 122 | 378 | 179 | 199 |
| 2500 → 5000 | 122 | 378 | 378 | 0 |

base 对训练端点的 199 个 dtype 变化均为 BF16 → FP32，转换后数值精确相同。
因此“文件或张量字节不同”不能直接解释为训练更新。

base → 5000 的模块分布：

| 模块 | 张量总数 | 数值变化 | 解读 |
| --- | ---: | ---: | --- |
| VLM | 345 | 0 | 下载 base 到最终端点数值完全一致，与冻结合同相符 |
| Action expert | 145 | 112 | 注意力/MLP 等有更新，33 个 norm 权重不变 |
| state/action/time projections | 10 | 10 | 全部出现净变化 |

33 个不变的 expert norm 包括 16 层各两组 norm 和最终 norm；三个端点均为
BF16 且每个值都是 1。这是需要下一阶段梯度/有效更新检查的线索。仅凭端点
不能判定它们没有梯度、被有意冻结、更新小于存储精度，或曾变化后返回原值。
本轮未构造模型，未验证运行时 requires_grad 或历史每步更新。

## 较大的相对变化

下面是 base → 5000 的 `||W_after - W_before||₂ / ||W_before||₂`。
它衡量整个张量的累计净变化，不是每个元素都变化这个比例，也不是故障贡献率。

| 参数 | 相对 L2 变化 |
| --- | ---: |
| `model.action_time_mlp_in.weight` | 21.4852% |
| `model.vlm_with_expert.lm_expert.layers.5.self_attn.k_proj.weight` | 15.9706% |
| `model.vlm_with_expert.lm_expert.layers.7.self_attn.k_proj.weight` | 15.4366% |
| `model.vlm_with_expert.lm_expert.layers.10.self_attn.q_proj.weight` | 15.2636% |
| `model.vlm_with_expert.lm_expert.layers.9.self_attn.k_proj.weight` | 15.0039% |

输出层 weight 为 5.0380%，state projection weight 为 3.8125%。
2500 → 5000 中最大相对变化仍是 `action_time_mlp_in.weight`，为 4.7367%。
不能按这个排名认定某层“训练坏了”：大变化可能有益，小变化也可能改变行为。

## 全量明细与可复算证据

- 登记计划：`configs/diagnostics/prepost-parameters-20260923-001.json`。
  SHA256 `1aa025dc28ec225b0f7d6c262747d98757e15b50a9f01eb2e75a449adb8ce1f5`。
- `.cache/prepost-parameters-001/comparison/base-to-2500.jsonl`
- `.cache/prepost-parameters-001/comparison/base-to-5000.jsonl`
- `.cache/prepost-parameters-001/comparison/2500-to-5000.jsonl`
- `.cache/prepost-parameters-001/comparison/result.json`：完整模块汇总、行文件
  SHA、资源与执行边界。SHA256
  `d6f2d50c29ca531773b57dbd5584f905de207f0e93f42a95f762cb0005055e86`。
- `.cache/prepost-verify-001/verification.json`：独立复算通过，三组各 500 行。
- `.cache/prepost-verify-001/verifier-executed.py`：与复算记录中的 SHA 完全
  匹配的实际运行源码快照。后续源文件只做一次换行修正以通过 Ruff E501；
  原始失败 lint 日志保留，未更改数值算法或重新解释证据。

每行含参数名/模块、shape/dtype/元素数、原始字节 SHA、有限性、均值、总体
标准差、RMS、L2、零值比例、数值/字节一致性、变化元素数/比例、差值
RMS/L2/最大绝对值、相对 L2 和余弦。零范数下的不可定义量为 null。
这些是 safetensors 中保存的全部张量；运行时未持久化 buffer 不在文件中，
也没有把参数文件冒称为完整运行时状态。

## 验证与资源

主计算用分块 NumPy float64 运算；独立复算用不同分块边界的 PyTorch
float64 运算，重新读取三个原始权重，核对 **1500 条参数对**的完整名称覆盖、
shape/dtype、原始 payload hash、统计值、差值指标、相等标记及总计。
数值复算使用相对 `1e-9`、绝对 `1e-12` 容差；字节及元素计数精确匹配。
两个进程均在前后重新核对登记的输入/源码 SHA。

小型测试 8 项通过，包括独立复算拒绝“统计值被修改后重新封存”的反例、
非 finite、缺失键/shape、零范数、BF16/FP32 数值一致但字节不同、符号零、
输入身份漂移与 create-only 输出。最终 Ruff 与 `git diff --check` 通过。
测试与 lint 输出分别保留在 `.cache/prepost-tests-001/`、
`.cache/prepost-verifier-tests-001/`、`.cache/prepost-final-lint-001/`。

两个真实文件阶段均由 WSL Bash 启动既有 digest 固定的 Linux Docker 镜像，
输入只读、无网络，实测 cgroup 为 2 CPU、4 GiB，runner 强制 600 秒；
主计算另验证额外 swap 为 0。主比较耗时 175.045 秒。
没有加载数据样本、构造 policy、模型 forward/backward、optimizer step、
模型下载或修改系统依赖。其他正在运行的容器未被停止或改变。

## 下一阶段与结论边界

现有检查点足够回答“哪些权重发生了净变化”，无需为这一问题再开一炉。
原始默认 processor 与 ALOHA 训练 processor 不同，下一阶段仍必须先核对
同一接口下的零更新初始化，再固定输入/noise/timestep 做推理与反向传播。
特别检查 33 个 norm 的 requires_grad、grad=None/真零、梯度和有效更新精度。
不得根据本报告直接改成 FP32 或改变学习率后重训。

已有 `scripts/iris_runtime.py::load_native` 是 CUDA/offline 限定入口，并会
读取模型与真实数据；本轮未调用。真实 trace 的独立 runner 尚未完成，
当前结果不能作为已经可直接运行完整 TorchLens/Gate 比较的声明。
下一次需要当前 AutoDL SSH 做只读环境/缓存/源码核验，先不必启动 GPU；
之后完成当前 runtime 下的 runner、source/config/input seal 和有限预算登记，
才进入 CUDA 固定输入采集。沿用前置报告中的 A/A、hooks、TorchLens 隔离
比较和保留失败证据要求，TorchLens 不接入现有正式 Gate 主循环。

这些差异没有证明原始策略能通过 Gate 4，也没有证明训练是失败的唯一原因。
原始模型新 Gate 3/4、激活差分、梯度差分、真实 step-0 reload 均未测。
canonical 004/005 的历史 Gate 4 0/5 与 M2 未完成状态保持不变。
