# Gaussian 噪声没有消除共同偏差，首动作关节反而退化

`zen-noise-transfer-20260911-001` 完成136次本地XPU forward。登记的“零噪声
是共同偏差主要来源”条件未通过。Gaussian可降低部分夹爪轨迹误差，但关节
没有稳定改善，平均首动作在两个物理单位组都退化。不能用更换噪声解释或
修复当前视觉泛化失败。

## 固定条件下的实测

同一Zen-uniform step316、45个相同首帧、50步输出、相同processor/合同。
三个seed使用固定CPU Generator标准Gaussian，逐次reset，与已有zero数组比较。
所有值为dev5标准动作MAE；关节radian、夹爪normalized。

| 噪声 | 完整关节 | 首动作关节 | 完整夹爪 | 首动作夹爪 |
|---|---:|---:|---:|---:|
| zero | 0.076904 | 0.022385 | 0.095305 | 0.014534 |
| 20260905 | 0.068685 | 0.030000 | 0.087077 | 0.015263 |
| 20260906 | 0.088270 | 0.032408 | 0.080950 | 0.018330 |
| 20260907 | 0.083850 | 0.059991 | 0.086129 | 0.013921 |
| 三Gaussian预测均值（附属） | 0.074633 | 0.027628 | 0.079798 | 0.013812 |

三个Gaussian的平均首动作关节MAE约0.04080，zero为0.02238；夹爪约0.01584，
zero为0.01453。均值ensemble不是独立噪声或训练重复，也未使完整关节/夹爪
超过train最佳常数0.050389/0.053093。不能挑20260905的关节或20260907的
首动作夹爪作为整体验收。

共同均值偏差平方：关节zero为0.008931，三个Gaussian依次0.004002、0.012314、
0.009691，未稳定低于zero的80%；夹爪zero为0.010537，三个Gaussian为
0.008293、0.004226、0.007327，均低于80%。夹爪完整MAE也下降，但首动作
非回退条件失败。两组完整条件不同时满足，整体判定false。

上一轮冻结的zero-train bias直接应用到Gaussian，未重新拟合：完整关节MAE
范围0.05914–0.06845、夹爪0.06564–0.07492，均不如各自最佳常数。
它能减少某些误差，但不能作为跨噪声稳定修复。

## 完整性、资源与限制

- 32项合成/回归检查、1项真实缓存检查、Ruff、格式和check_env通过。
- 模型前45图SHA、state/action/task、缓存、artifact、VLM依赖清单与原生source
  SHA通过。首个样本的新零噪声完整50步输出与历史逐值相同；只作这一个样本的
  独立进程模型重载控制，未声称45个zero样本全部重跑。
- 3×45新Gaussian加1个zero控制，共136次forward。500个模型参数张量
  前后摘要全部一致，无optimizer、权重更新或训练。新预测数组exact reload通过。
- 既有Linux Docker / PyTorch2.11.0+xpu，Intel Graphics [0x7dd1]，网络关闭，
  2CPU、主存/memory+swap各6GiB；XPU peak allocated 1,268,320,768 bytes，
  低于4GiB限制。采集/参数审计约181.435秒，低于600秒预算；未测主存峰值。
- 既有torchcodec加载警告后由PyAV解码。配置反序列化提示
  `Device 'cuda' is not available. Switching to 'xpu'.`；runner预先要求XPU，
  loader在模型构造前显式指定XPU，实际forward也在XPU，没有尝试CUDA或改设备方案。

原始证据 `runs/zen-noise-transfer-20260911-001/`，预检与模型日志
`runs/zen-noise-preflight-20260911-001/`。完整参数摘要和逐维指标保留在原始
JSON，tracked同名JSON保留精简指标及原件SHA，不丢弃失败噪声条件。
这是历史Zen的frame-0诊断，不能外推到coverage40 B、非初始state或闭环。
dev5已经参与开发，hidden封存；无SSH、新训练、Gate或发布，M2未完成。
