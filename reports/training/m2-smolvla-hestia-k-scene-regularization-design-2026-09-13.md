# K侧定向修复设计：训练时约束图像K场景残差

设计ID `hestia-k-scene-regularization-design-20260913-001`，**launchable=false，训练未授权，尚无修复效果**。这是用户批准的“完成定位后给出定向修复方案”交付，不是下一炉执行许可。

## 根据证据选择的方向

Q重放结果表明，640固定Q保留91.92%的K均值左关节MAE收益，且自然Q反馈进一步增强收益。结合前一轮保留场景V优于抹除V的结果，优先约束图像K的场景依赖，同时让原生flow目标保留任务所需的信息。直接K均值替换损伤训练拟合及首动作夹爪，不能作为完成修复。

**选定的候选是软约束，不是已确认的底层修复。** 跨场景K变化中也可能含有有用信息；本实验检验在训练目标中小幅抑制该变化能否改善映射，而不是假定所有场景差异都应消失。K权重与上游表征的唯一成因仍未区分，后期V损伤不在同一炉修复。

## 单一实验轴与实现合同

未来在v2训练harness添加默认关闭的`image_key_scene_regularization` feature，通过现有`TrainingFeature.install/restore`和有序feature registry安装。旧训练入口、依赖缓存及本轮推理collector保持原身份。

仅在训练时捕获八个奇数cross-attention层真实图像0:64位置的K投影输出，保留梯度；V、Q和完整attention计算保持原生。记本batch的K为`K[b,p,c]`，按相同token位置跨batch场景求均值。每层正则项为：

`R_l = sum((K - mean_batch(K))^2) / ((B - 1) * P * C * s_l)`

`s_l`是新鲜原始基座在train40上的同类无偏场景方差，标签无关、仅校准一次并封存，不从Hestia640/1280权重或开发场景取得。`s_l <= 1e-12`或非finite时拒绝启动，不用任意回退值。训练时在线计算K，不复用训练中逐渐失效的K特征缓存。batch1仅用于原生forward前检；正式正则化与optimizer smoke要求batch4。

在原生policy完成有效元素归约后，加入标量`L = L_flow + lambda * mean_l(R_l)`，每次训练forward只计入八层各一次；不能按50个动作槽或重复调用重复累加。推理/validation/export路径不安装正则化hook，也不替换K。记录原生flow loss、正则项、总loss及各层K梯度/更新，避免只看总loss掩盖拟合损失。

固定对照`lambda=0`、候选`lambda=0.01`，不做看到dev后调lambda的搜索。0.01是本设计选择的小幅实验值，未经有效性验证。lambda=0必须完全旁路新增运算，原生输出、梯度和RNG均精确一致。参数可训练范围、VLM冻结、Q/V学习规则均沿用原Hestia，不同时加K-only优化、dropout、EMA、早停或V回滚。

## 新比较的默认规模与验收

两个新鲜基座独立初始化臂；同train40 frame0、batch4、seed20260809、1280updates、5120exposures，warmup16/cosine1280、AdamW等来自Hestia主实验的SHA绑定配置。历史Hestia只作背景，不能替代新lambda=0对照。保存320/640/960/1280；固定1280为候选终点，640仅作诊断，不按本次dev观测重新选保存点。

计划中的未来梯度工作为两臂各2步隔离smoke加两臂各1280步训练，共2564步；smoke状态不进入正式臂。拟定单独计算上限60分钟、受保护关机上限70分钟，GPU allocated8GiB/reserved10GiB、RSS8GiB。该预算未实测且未授权；前检计时预计不能容纳时不启动正式臂，不自动扩大预算。固定噪声评估、独立reload、数组回传均需纳入新执行登记。

实现验收先覆盖：lambda=0原生精确等价；正则项和解析梯度；跨batch重排不变性；不压平共享token模式；仅真实图像位置贡献；B<2、缺层、重复调用、非finite和统计身份漂移拒绝；异常后恢复；eval无hook。再验证真实缓存、normalization、batch1原生forward、batch4两步smoke与独立reload。

科学候选验收固定为：两臂原生完整评估成功；候选dev5左关节完整MAE/MSE在四噪声及逐场景删除下均优于新对照；train40拟合仍通过原协议；四物理组在首动作及完整窗口的标准空间MAE/MSE均不回退，并击败train mean/median常数。任何一项失败记负结果并停止，不用平均左关节收益覆盖夹爪退化。dev仍是开发集，hidden封存；通过这些条件也不能宣称独立泛化或M2完成，Gate3/4另行登记。

## 当前未满足的前置条件

本轮Q重放的float64/BF16一ULP附加验收失败，虽然动作指标、调用身份和保存张量哈希通过。必须先单独处理这个数值证据缺口：从已封存320个样本出发核对真实算术阶段、舍入与后端合同，保留原失败，禁止用放宽原阈值制造通过。若需要原CUDA上的张量级验证，须另登记预算与身份，不计为现有2160次授权的延续。

新feature尚未实现；其源码、校准统计、两臂配置、阶段命令、资源实测和训练授权均未封存。本设计明确这些实现决策和候选标准，但不是可执行配置。当前没有训练或部署动作，修复状态为`proposed_not_validated`。
