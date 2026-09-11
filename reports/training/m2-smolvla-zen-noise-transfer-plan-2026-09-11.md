# Zen 完整轨迹的推理噪声单轴复测

登记 `zen-noise-transfer-20260911-001`。用户要求连续推进排查；本轮为有界
本地XPU推理诊断，不是训练或远端计算。现有偏差分析只有零噪声完整数组，
本轮检验共同偏差是否主要来自零噪声选择。

固定原Zen-uniform step316 artifact、模型/processor/revision、Action Contract、
相同40 train / 5 dev首帧和完整50步目标。只改变推理噪声：固定CPU Generator
seeds `[20260905,20260906,20260907]`，生成原生 `(1,50,32)` 标准Gaussian，
按原batch state dtype转换到XPU。每个seed在所有场景使用相同完整噪声，
每次policy.reset，eval/inference_mode、原BF16 autocast不变，不加cache或hook。

模型前完整检查45个image SHA/state/action/task、固定缓存、artifact和VLM
dependency inventory以及原生source SHA。第一张图先做零噪声完整chunk
逐值复现旧输出，失败就停止；这只验证一个首帧的模型重载控制，不能冒充
45张图都独立重跑了零噪声。之后3×45 Gaussian forwards，共最多136次。
模型全部参数前后摘要逐一核对，禁止optimizer和训练。

主对照为旧45图零噪声输出与3组新Gaussian输出。每组单独报告full、first、
early、中段、late和last的关节/夹爪MAE、常数、错配收益及均值偏差。
控制和处理均按相同Action Contract bounds评价，原始输出保存。
附带报告三个Gaussian预测的均值ensemble，以及上一轮已冻结的zero-train
bias应用到每组输出的结果；后两者只作附属诊断，不重新拟合bias、不选最佳seed。

支持“零噪声是共同偏差的主要来源”须两组均满足：全部3个Gaussian的full
均值偏差平方低于zero的80%；3组平均MAE低于zero；平均首动作MAE不退化
（1e-8容差）。条件不全满足，保留负/混合结果，不重新选噪声或放宽阈值。
三个噪声不是三次训练重复，ensemble也不是新增独立样本。

本地既有Linux Docker/WSL Bash/XPU，固定原镜像digest；网络关闭，2CPU，
内存和memory+swap各6GiB，XPU allocated上限4GiB，分析上限600秒，最多
136 forward。运行前check_env、Ruff、合成噪声/RNG及已有dummy hook检查、
真实缓存测试；之后模型前身份检查通过才加载权重。每5个场景记录进度。
身份/zero parity/非finite/内存/时间/参数不变检查失败即停止，保留失败日志，
不自动切设备或改变精度。输出create-only、数组exact reload。

这是现有Zen artifact的离线推理比较，不代表coverage40 B或vfunfreeze。
dev5已参与开发；不改权重、训练、Gate、远端实例或公开内容。M2未完成。
