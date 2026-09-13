# Hestia Q/attention 重放：主因定位执行登记

登记 `hestia-q-replay-20260913-001`。用户批准先定位再形成定向修复方案，选择640/1280两保存点，并明确授权自行通过Chrome开启原AutoDL实例、SSH推理、回传与受保护关机。随后要求实施完整计划。本轮不训练、不选checkpoint、不执行Gate、不打开hidden，也不提交或推送。

依据同日期scene-kv完成报告：640早期左关节差距更强地指向K侧场景依赖/QK匹配，V场景内容仍有用。此诊断区分固定原生Q后K干预效果与后续Q反馈，不将K均值或原生重放当作部署修复。

固定train40/dev5顺序、历史四噪声、batch1、50槽、十去噪步。每保存点按noise、scene串行执行六模式：native、qself、aself、kmean、kmean_ablock、kmean_qnative。原生参考严格限定同保存点、scene、noise、去噪步及expert奇数层。每保存点1080次policy forward，共2160次、0optimizer；其中1800次输出必须精确匹配原生或历史K均值数组，全部控制通过后才接受剩余360次固定Q干预证据。失败立即停止，保留部分计数与异常；不自动重试。

原生eager源码SHA为`996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf`。实例级wrapper仅观察expert cross-attention；前缀prefill和self-attention走原方法。Q在原生RoPE后重放；attention概率在原生softmax及dtype转换后、乘V前重放。记录原生Q、实际Q、K/V、mask、概率及输出的dtype/shape或摘要。完整241-token GEMM、真实图像0:64之外的输出和全部V保持原生。K均值复用已封存的标签无关train40银行；train使用train39排除自身。保存每次投影的原生输入/输出SHA并核对旧银行。

原生模式同时调用原生eager与按原顺序提取概率的实现，逐值相等才保留参考。每场景仅保留当前noise的80组Q/概率，不累积所有重放张量。每保存点row0/row40、noise0的全部80层步保存完整Q/K/V、mask、概率、attention输出，合计320个sentinel文件。另保存所有模式的动作数组、调用摘要、模型参数不变及输入证据。回传文件总上限512MiB，归档513MiB。独立CPU sentinel复算使用float64及显式BF16舍入，容差为一个BF16 ULP；该近似算术检查与CUDA输出零容差门禁分开报告，绝不用于放宽CUDA控制。

科学主指标为每保存点standard/dev5/full/left_joint的MAE和MSE，要求四噪声及逐场景删除均改善才称固定Q效果一致。kmean_qnative相对native给出固定Q路径效应；kmean相对kmean_qnative给出释放Q反馈后的增益/抵消，不能当作独立可加因果份额或证明Q权重错误。必须同时保留两空间、两split、首动作/完整、四物理组、train mean/median基线、均值偏差/场景方差/协方差。训练常数按自身排除。dev反复参与诊断，四噪声不是独立训练。

本地所有数值/ML检查经WSL启动固定离线Docker，2CPU/2GiB；已完成174项相关测试及Ruff/格式检查，日志位于`runs/hestia-q-replay-preparation-20260913-001/`。远端使用既有AutoDL CUDA例外、已缓存权重/数据和新内容寻址工作区；不嵌套Docker、不修改旧工作区。先准备receiver和守护器，doctor、normalization绑定、CPU和全部源码/输入SHA验证均在权重加载前通过。worker总上限4200秒，每保存点最多2100秒，保护截止4800秒，CUDA10GiB/RSS8GiB；停止条件包括身份漂移、非法/非finite动作、控制不等、预算/资源超限和其他worker冲突。预计沿用历史推理约30至50分钟，但现场原生控制计时才是实际依据。

完成后先保存退出状态与manifest，回传文件集合/类型/路径/大小/SHA核验及匹配收据，然后按已授权流程关机并Chrome复核；回传阻断也不超过保护截止，不释放实例。不为补旧shutdown文件额外开卡。本轮输出定位结果和下一轮单轴修复登记：明确证据边界、具体干预位置、保留的V信息、是否需要训练、预算与验收；结果混合则保留未区分结论，不强行关闭唯一根因。当前科学结果、Gate3/4均为not measured，M2未完成。
