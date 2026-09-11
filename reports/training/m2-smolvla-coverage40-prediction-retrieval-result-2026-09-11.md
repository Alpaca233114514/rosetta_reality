# coverage40 A/B 完整预测已取回，历史指标逐项重现

用户提供当前SSH连接后，按已公开计划只读核验并取回19个普通文件，
总计5,515,092 bytes（约5.26MiB）。旧评估工作区commit仍为
`02cbf63269c2f3038e9e2cb7ec1af64d84b17813`。路径/类型、总大小、五个
历史小文件以及14个数组的文件SHA均通过；远端没有文件修改或GPU计算。

现有离线Linux Docker中，`scripts/verify_coverage40_predictions.py` 完成
两份完整bundle的文件SHA、数组SHA、schema/shape/dtype/finite检查，并核验
A/B模型、四组噪声、physical/execution contract、共同非视觉条件和历史reload
回执。随后原 `compare_arms` 重算558个数值字段和199个其他字段/决策，
全部匹配；最终完整 `recomputed.json` 文件SHA也与历史 `comparison.json`
完全一致：`cda788c7ddee3771d5923ef4c82844bac1077248dffe289212da801e74756f50`。

因此原A/B开发验收0/4的负结果被本地保存证据复算确认。不是新平台上的
模型forward复现，也不是新的独立进程模型reload；没有重新加载checkpoint。
历史CUDA资源峰值没有重新测量。

- Linux Docker镜像digest沿用计划；2CPU、主存/memory+swap各2GiB、网络关闭。
- check_env、Ruff、格式及34项既有bundle/比较回归通过。
- 保存七组完整数组，每臂预测shape为4×45×50×14，目标45×50×14；无hidden。
- 原始取回目录 `runs/coverage40-prediction-retrieval-20260911-001/`。
- 控制清单、执行计划和日志 `runs/coverage40-prediction-retrieval-control-20260911-001/`。
- 本地复算证据 `runs/coverage40-prediction-verification-20260911-001/`。

第一次本地启动引用了不存在的Git Bash安装路径，在SSH前失败；随后定位
现有安装完成连接，没有安装依赖或改变认证/主机校验策略。所有SSH会话已退出。
未改变实例电源状态，当前计费状态未核验；没有训练、权重传输或Gate，M2未完成。
下一阶段可用已验证的实际B数组检查偏差和时移，不再用Zen结果代替B。
