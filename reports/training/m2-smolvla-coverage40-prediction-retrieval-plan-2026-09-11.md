# coverage40 A/B 历史预测取回计划

状态：本地准备完成，SSH 与传输尚未执行。目标是直接检验实际 A/B 的共同
偏差、时移和首动作误差；历史 Zen 对照不能替代 B 的模型级证据。

历史权威入口为 `m2-smolvla-native-visual-coverage40-result-2026-09-10.json`。
完整模型和预测留在远端，本地现有摘要不含七组原始数组。此次只取已有预测，
无需加载模型、运行GPU或下载checkpoint。不会续跑 Hestia 或其他训练。

## 具体传输范围

远端目录相对已核验的durable root：
`dev/visual-hermes-post-20260910-02cbf63/runs/visual-hermes-coverage40-eval-001/`。
本地新目录：`runs/coverage40-prediction-retrieval-20260911-001/`，已存在则停止。

只允许19个普通文件：

- `execution-contract.json`；
- `A-reload-check.json`、`B-reload-check.json`；
- `A-first/manifest.json`、`B-first/manifest.json`；
- A-first与B-first下各自的 `internal_grippers.npy`、`noise.npy`、
  `normalized_predictions.npy`、`normalized_targets.npy`、
  `standard_predictions.npy`、`standard_targets.npy`、`valid_mask.npy`。

先只读核验当前实例身份和durable root、文件路径/类型/字节数及manifest SHA。
拒绝符号链接、非白名单路径和根目录逃逸。总字节上限32MiB；精确大小需现场
清点，超限停止。先取manifest并核对历史SHA，再依据其中文件SHA接收数组；
每个目标文件create-only，保留部分下载与失败记录，不覆盖或删除远端原件。
不传输权重、optimizer、图片、数据集或完整日志，不修改实例电源/计费状态。

历史小文件身份：

| 文件 | bytes | SHA-256 |
|---|---:|---|
| execution-contract.json | 9676 | 314abd279934dd4e1ef762eb5e8299ed0601b8b14857aec9fd2438fa5f109801 |
| A-first/manifest.json | 30392 | 8c3c413e885ef2631c0a360b86c061509f12b3dbeaa079c3ace7c26496cde2ea |
| B-first/manifest.json | 30396 | b57252f9741feeec4d836a223b4e30e2301e722b80482a83f2ec2d36f3dc117b |
| A-reload-check.json | 318 | 03c13e2a7f3f07f4b33a505537c02d30f9d4e0793ef791da693df3529b5a3690 |
| B-reload-check.json | 318 | 03c13e2a7f3f07f4b33a505537c02d30f9d4e0793ef791da693df3529b5a3690 |

## 取回后验收

先在现有离线Linux Docker中调用 `visual_coverage.read_bundle`，验证七组
数组的文件/数值SHA、dtype、shape、finite、45场景、hidden未加载及合同。
核验A/B模型SHA、四个噪声、图像/nonvisual hashes与历史身份；通过原
`compare_arms` 重算并与既有小型指标对照。这里是保存证据的读取复核，
不是新的独立进程模型reload，也不是在新平台重现CUDA推理。

身份或原始指标无法复现即停止归因分析。通过后另行封存B诊断输入身份，
沿用预先公开的train-only均值偏差校正和全局lag[-10,10]选择方法：各噪声
分别保留、train40校正LOO、dev5不选超参、不挑最佳噪声；同时报告关节、
夹爪、首动作与完整chunk、两个train常数和正确图收益。不会把dev当独立测试。
没有新模型forward、权重修改、optimizer、Gate或任务成功声明。

当前需用户提供本次可用SSH连接信息，并明确开放上述只读清点与最多32MiB
传输范围。此前Hestia交接停在SSH门前；旧端口和关机记录不证明当前实例状态，
不自动连接、开机、租卡或复用旧计算预算。计划本身不构成执行授权。
