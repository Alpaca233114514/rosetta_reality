# 本地视觉泛化排查：当前结论与实际B复查入口

当前没有找到可接受的视觉泛化修复。已完成的本地诊断约束了若干解释，但
主要使用历史Zen-uniform step316，不能替代实际coverage40 B的测量。
下一项有判别力的工作是取回历史A/B完整预测进行同口径复查；取回计划已
固定文件白名单、SHA和32MiB上限，不需要GPU计算或checkpoint传输。

## 已确认的结论

| 问题 | 已有证据 | 解释边界 |
|---|---|---|
| 更深KV能否修复读出？ | late层未优于early层；扩大train-only正则搜索仍未整体超过常数 | 不足以归因于探针层或单个alpha选择；不是模型能力上限 |
| 图像中的socket位置能否读出？ | early KV二维socket质心dev MAE约4.38/5.71像素，低于常数12.43/12.91 | 支持位置线索可访问；不证明3D几何、完整任务信息或policy利用正确 |
| 位置对较晚动作是否更有解释力？ | 左臂末步相对常数的误差比改善到0.816，但登记dev<0.8未通过 | 有时间相关差异，尚不足以解释动作泛化 |
| 比较标签是否一致？ | raw gripper越界在投影后改变；原生训练已做该投影 | 修正诊断比较口径；没有发现训练缺投影的bug |
| 线性完整chunk读出能否胜过基线？ | dev关节/夹爪MAE0.05316/0.06268，优于Zen原生但劣于常数0.05039/0.05309 | 更低误差不等于视觉验收通过 |
| 共同偏差校正是否足够？ | train-only偏差校正降低完整误差，关节dev及夹爪LOO仍失败 | 不能部署为已验证修复，首动作还有回退 |
| 零推理噪声是否主要问题？ | 3固定Gaussian seed共135次forward加1次zero控制，未稳定降低两组偏差；平均首动作退化 | 不能用换噪声修复；无权重更新 |
| 全局时间错位是否足够？ | train及40次LOO一致选lag=-7；完整误差降低>20%但两组不如常数，首动作逐值不变 | 不能证明数据错标，不能改善当前只执行首动作的策略 |

以上每个结论均有独立计划、原始证据、结果JSON/Markdown与提交。入口依次为
`m2-smolvla-contextual-kv-depth-result`、`m2-smolvla-kv-regularization-result`、
`m2-smolvla-socket-position-result`、`m2-smolvla-socket-action-horizon-result`、
`m2-smolvla-kv-full-chunk-result`、`m2-smolvla-chunk-bias-result`、
`m2-smolvla-zen-noise-transfer-result`、`m2-smolvla-chunk-time-shift-result`，
均位于 `reports/training/`，日期后缀为 `-2026-09-11.{md,json}`。

这些结果支持优先调查视觉线索到动作轨迹的映射及拟合强度；还不能定位到
某一层的确定故障，也不能排除未测因素。不能把“socket位置可线性读出”
升级为“视觉编码器完全没有问题”。

## 实际coverage40 B的独立事实

历史B的train40关节拟合4/4、夹爪1/4；dev关节/夹爪均0/4。首动作两组
MSE相对A回退。该事实来自2026-09-10原始报告，而非本地Zen推断。
四条件为固定推理噪声，五开发场景已参与开发，不是独立测试；hidden未打开。

本地未找到该B权重和完整A/B七数组；只存在历史小型摘要。新的取回范围见
`m2-smolvla-coverage40-prediction-retrieval-plan-2026-09-11.md`。在manifest与
原口径复算通过后，才将同一train-only偏差/时移方法应用到实际B；不利用dev
选择方法、lag或噪声。不需要继续增加Zen后处理试验来代替这一步。

## 后续执行状态

Hestia的197项本地合成回归和5120样本顺序已通过，三文件格式修正保持AST
不变；见 `m2-smolvla-hestia-local-regression-2026-09-11.{md,json}`。
完整AutoDL验收仍待现场执行，Hestia主训练未运行。

取回预测需要本次可用SSH信息和明确授权。现有授权未跨过上轮SSH边界；
没有连接、开机或启动新计算预算。当前无模型/训练进程等待收尾。
任何新模型实验继续遵守既定预登记和授权门禁；本轮闭环成功未测，M2未完成。
