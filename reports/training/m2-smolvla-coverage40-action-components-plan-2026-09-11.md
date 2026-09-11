# A/B动作分量误差分解预登记

登记 `coverage40-action-components-20260911-001`。已有原始bundle身份和原口径
重现均通过；本轮只描述保存预测在哪一动作组/时段失败，不拟合校正器、
不改变模型或验收门槛。目的是定位B夹爪训练拟合不足，以及开发误差的构成。

固定A/B两臂、四个噪声、train8/train40/dev5、normalized/standard两空间。
沿用full/first/early/middle/late/last窗口。按合同名称和单位拆分左/右关节、
左/右夹爪，同时保留全关节/全夹爪以对照历史汇总；不是按结果选择维度。
A的train40包含32个未训练场景，不能当作全部训练拟合。

每组/窗口先按场景去均值，计算target variance、prediction variance、
covariance和共同均值偏差平方，检验
`MSE = target_variance + prediction_variance - 2*covariance + mean_bias_squared`。
同时记录MAE、预测/目标场景方差比、场景相关系数、可部署train40 mean/median
基线。方差为零时相关系数标为null，不制造数值。所有噪声分别保留，四条件
均值仅作描述，不作独立训练重复。每维full误差用于确认聚合恒等式。

没有新的选择阈值或通过门槛；结果只支持后续实验排序，不声称发现确定训练
bug或已修复泛化。normalized误差不能直接作为真实动作/任务误差。
首动作与后续槽位分开解释，场景方差压缩不等于证明视觉编码器失效。

在现有离线Linux Docker执行，2CPU、主存/memory+swap各2GiB、180秒。
先源码Ruff/格式与分解恒等式检查，输入/源码SHA冻结后一次执行。
输出create-only；不加载原始样本、checkpoint或模型，不执行SSH、optimizer
或Gate，hidden不进入输入，M2未完成。
