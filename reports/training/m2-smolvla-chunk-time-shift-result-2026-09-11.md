# 全局时移降低轨迹误差，但没有解释首动作失败

`chunk-time-shift-20260911-001` 完成。40个train样本共同选择lag=-7，
即输出使用提前7个槽位的预测值（-140ms），40次校正LOO也全部选择-7。
统一时移有可重复的离线关联，但登记的充分解释条件未通过。

## 实测

下表为相同Zen-uniform step316、45个首帧、50步chunk上的dev5 MAE。
关节单位radian，夹爪normalized；两组分别比较，不混合解释物理量。

| 条件 | 完整关节 | 完整夹爪 | 首动作关节 | 首动作夹爪 |
|---|---:|---:|---:|---:|
| 原始zero预测 | 0.076904 | 0.095305 | 0.022385 | 0.014534 |
| train选定lag=-7 | 0.055859 | 0.073252 | 0.022385 | 0.014534 |
| 最佳train常数（完整chunk） | 0.050389 | 0.053093 | — | — |

完整误差两组都下降超过20%，正确图片收益分别为0.004420/0.006010，
但均未超过train常数。校正LOO关节0.052315低于两个常数，夹爪0.070446
仍高于两个常数0.044788/0.039681。`global_shift_supported=false`。

负lag在边界保持首个预测，7个槽位使用端点；所有目标槽位仍参与评分。
首动作数组逐值不变，因而不能改善当前receding-first-action执行。
按组分别查看的train最优lag为关节-7、夹爪-6，仅作诊断，没有分组调参后
应用到dev。结果不能证明数据存在时间错标，也不是已部署的policy修改。

## 验证与范围

复用已有native/projected-target数组，输入和代码SHA均通过。
固定Linux Docker离线CPU运行，check_env、Ruff、格式检查及9项合成/回归
检查通过；CPU测试的既有XPU设备数为零警告不影响测试。
新数组exact reload通过。没有新模型forward、原始数据读取、训练或SSH。

完整grid分数、40个LOO lag和逐维指标保留于
`runs/chunk-time-shift-20260911-001/`；预检/执行日志在
`runs/chunk-time-shift-preflight-20260911-001/`。同名tracked JSON保留精简
指标和原始证据SHA。dev5已参与开发，hidden保持封存。
本结论仅针对历史Zen首帧，不是coverage40 B复测；未执行Gate，M2未完成。
