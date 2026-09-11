# 已拟合C的训练到开发响应迁移分解

登记 `hestia-transfer-decomposition-20260911-001`。Hestia主实验已通过回收证据
逐字节复核：C train40全组4/4、dev5全组0/4。现在只描述B→C误差变化
由哪些项组成，不再以B尚未拟合解释C的失败，不生成新训练或校正。

固定使用同一次Hestia的B-first/C-first七数组，四噪声、train40/dev5、
normalized/standard两空间，full/first/early/middle/late/last六窗口。
组为全部关节、全部夹爪，以及左右关节/夹爪；沿用已验证的场景中心化分解。

每组同时计算B/C的MSE、MAE、目标/预测场景方差、协方差、均值偏差平方、
场景相关，以及train40 mean/median常数；验证两臂目标方差相同，且
`C_MSE-B_MSE = delta_prediction_variance - 2*delta_covariance + delta_mean_bias_squared`。
四条件逐项保存，均值/min/max只作描述；不按dev选择噪声、阈值、维度或窗口。

重点区分训练拟合时恢复的场景响应，是否在开发场景得到足够正确的对应；
方差或协方差改变不是某一网络层故障的证据。该分解不证明数据不足的唯一
原因，也不验证解冻/几何监督等后续方案。首动作单列，不用完整chunk结果
替代当前执行策略的验收。

既有离线Linux Docker、2CPU、主存/memory+swap各2GiB、180秒上限。
先源码Ruff/格式和输入SHA检查，复用已通过的分解反例；原件不改，输出
create-only。无模型/图像/原始数据加载、optimizer、SSH、Gate或文件清理。
dev仍是开发集、hidden未打开，M2未完成。
