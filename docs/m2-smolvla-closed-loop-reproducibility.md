# 闭环复现性：配对诊断入口

本入口核对同一 canonical step-5000 在固定 reset、噪声和执行合同下的重复性。
不改变模型、processor、BF16、随机数规则、确定性开关、动作或正式 Gate 阈值。
原 004/005 仍为 0/5。代码和合成测试不代表真实闭环复现性已通过。

## 三臂与身份

新增 `scripts/diagnose_smolvla_gate.py collect-repro --plan ...` 和
`compare-repro --baseline ... --repeat ... --traced ... --output ...`。
`verify --directory ...` 支持采集包的独立结构与数值核验。

每臂必须在独立 CLI 进程、独立新注册和新目录执行：

| role | 行为 |
| --- | --- |
| baseline_a | 原 Gate 引擎，加相同的模型边界与积分状态采集 |
| baseline_b | 精确重复 baseline_a 的观测方式 |
| full_trace | 相同采集外，再加既有完整 rollout trace |

三臂的 `reproducibility.pair_id`、seed、checkpoint、processor、代码清单、
动作合同和推理设置必须相同，只有 role、run_id、输出目录和进程身份等执行元数据
不同。seed 必须预先从原 1000–1004 中选定，最多 500 步、50 Hz、每次只执行
首动作。一个三臂 campaign 只检验这个 seed，不构成五 seed Gate 验收。

`configs/vla/closed_loop_reproducibility_001.json` 是不可执行的 baseline_a 草稿。
真实运行需创建三份新注册、补齐全部文件 SHA 与既有端点/G3 验收身份，再分别
声明新的 `execution.stages=["collect-repro"]`、授权窗口、GPU/库版本和 watchdog。
本地编辑授权不代表 SSH、GPU 或 rollout 授权；CLI 不开机、不连接 SSH。
既有 Seed 3 命令不能使用这些计划，原 Seed 3 draft 保留不变。

## 记录内容

- `reset/`：初始 observation 和 MuJoCo `mjSTATE_INTEGRATION`。
- `predictions/NNNN/`：真实 observation、instruction、processor batch、完整
  noise、normalized/internal/decoded/projected 全数组。包括 BF16 原始位型。
- `steps/NNNN/before/` 与 `after/`：原执行命令、积分状态、下一观测、reward、
  success、terminated、truncated。写盘失败保留已经落盘部分并停止。
- `runtime.json` 与 `runtime-final.json`：Python/库版本、GPU UUID、驱动、
  已加载 CUDA/cuBLAS/cuDNN/MuJoCo/EGL/GL 库 SHA、实际 PyTorch 确定性与
  attention 开关、相关环境变量。只读取，不设置或修正这些开关。
- 编译后的 MuJoCo 模型通过内存 MJB 序列化求 SHA，结束时复核模型未变。
  积分状态含 warmstart，读取不调用 `forward`、`setState` 或额外仿真 step。

采集复用现有 8 GiB RSS、8/10 GiB CUDA、截止和 watchdog 门禁；证据预算最高
4 GiB/臂，写入逐步检查并预留收尾空间。没有自动重试、训练或 checkpoint 选择。
真实物理接口缺失时停止并保留 incomplete，不用零状态替代。runtime 身份不足
时比较返回 insufficient_evidence，不能记为通过。

## 比较与解释

先校验每个包的完整文件清单/SHA，再核对逐步连续性、执行首动作、终止、
完整 trace 与采集记录的一致性；之后分别比较 A/A 和 A/full_trace。
比较顺序为 reset → 每步积分状态 → observation → instruction → batch → noise
→ normalized → internal → decoded → projected → executed → 下一状态/观测/结果。
保留 dtype、shape 和逐位差异，包括 signed zero；不以 MAE 接近替代精确一致。

- `matched`：所观测边界一致。synthetic 结果始终不能宣称真实 rollout 通过。
- `diverged`：报告首个不一致 step、边界及字段；不是唯一根因裁决。
- `incomparable`：运行环境、源码、端点或配对身份不同，应先恢复比较条件。
- `insufficient_evidence`：包不完整或运行身份不足。

非 matched 比较返回非零退出码。输出 create-only，必须在源证据目录之外。

**两条基线也有观测开销，因此通过不等于完全无插桩的轨迹等价。**
它回答的是增加旧 full trace 的增量影响。若 A/A 本身已分叉，不能据 A/full_trace
分叉认定 trace 有副作用。

`mjSTATE_INTEGRATION` 不包含所有 Python wrapper、任务 RNG、callback 状态；
本模块没有实现物理状态恢复，也不宣称可从任意保存步重启得到同一轨迹。
跨驱动/渲染实现的完全复现、真实 CUDA、完整五 seed 及 M2 均待实测。

## 本轮依据

只读复核发现：004/005 的四份 `collect-first/arrays.npz` 与
`collect-reload/arrays.npz` SHA 相同；1082 个共有源文件 hash 一致；inference
和 Gate 协议文本一致。但旧 004 无逐步 trace，不能从旧包反推首次分叉。

相关一手资料：

- [PyTorch 2.8 reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html)：
  seed、算法选择与非确定性计算需要分别检查。
- [cuBLAS 12.8 reproducibility](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#results-reproducibility)：
  注意版本、架构及并发 stream 条件，不能仅凭型号判断。
- [MuJoCo integration state](https://mujoco.readthedocs.io/en/3.6.0/programming/simulation.html)：
  warmstart 及用户输入属于积分状态；接触可放大小差异。
- [LeRobot LIBERO issue 4152](https://github.com/huggingface/lerobot/issues/4152)：
  相同 seed 不保证实际初始状态配对的案例；不是本项目 ALOHA 已证实的故障。
