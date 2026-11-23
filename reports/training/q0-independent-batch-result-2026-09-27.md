# 独立检查整批结果及关机策略修订

身份：`q0-independent-checks-20260927-001`。用户授权把所有可实现的独立检查合并，
并进一步明确“所有实验做完才关机”“按最新的做”。因此旧单批次结束关机策略
不再适用于后续整体实验：前置检查完成、单项失败或技术阻塞均不得自行请求关机。
每项工作的超时、资源与安全停止条件仍须保留。平台定时由用户管理，本代理未核验/取消它。

## 已实际完成

新工作区相对数据根为`workspaces/20260927T113008Z-66a5bbcdbdcd-4a3afe5ff806`。
52个上传文件的SHA全部匹配；计划SHA
`c27d233b9f5234a46f8abc54b0284847018cac0f8cacd5014c4c1e84c00a64a9`。
部署前28项标准库回归通过；worker内46项测试及18个subtests通过。
两组存在重叠，不相加作为独立测试数量。

五阶段全部退出0：环境、合成测试、证据审计、状态恢复、拒绝边界。

- 证据审计独立核验上一轮127文件/4,663,991 bytes、注册与代码/协议绑定、
  G1标准算术和benchmark估算，确认G0通过、G1失败后G2未开。
  此为保存证据复核，不是物理结果重演。
- 状态恢复：固定seed10、总计6个保持动作，两段各3步。
  完整mjSTATE_INTEGRATION与已识别Python wrapper/RNG状态恢复后，
  三个对应步骤的物理状态、观测、reward/done/contact及安全字段精确一致；
  模型物理定义hash未变。仅证明reset附近这三步，不证明带载接触阶段可恢复。
- 输入边界：32个固定case，各两次重建，所有工程判据通过，
  negative输入没有静默生成动作。使用stub solver，不证明真实IK或正式G4。

没有新model forward、optimizer update、G1重试、恢复采集或训练。
provider_qualified仍为false，三项模型实验整体没有完成。

## 回传与用户关机更正

新证据已回传到`.work/q0-independent-checks-20260927-001-received/extracted/`。
19个原封存文件/195,798,020 bytes全部重新计算SHA和大小，0不匹配；
manifest SHA为`fbfe939b00969a96022abc38abb8434b5cf96898c588bc9b6fffefc9bb40840a`，
worker-exited绑定一致。主要空间来自完整数值状态/观测，不进入Git。

原worker PID1812已经完成。用户更新关机要求后，先精确绑定guard PID1811、
start_ticks与任务argv，发送SIGSTOP防止进入关机；随后核验状态确为T、
worker已完成，再终止该旧guard。远端`user-shutdown-cancelled.json`于
2026-09-27 11:36:54 UTC记录guard_live=false、shutdown_request_exists=false。
没有终止计算中的worker；当时已经没有本批计算任务需要独立deadline执行。
旧registration/guard-armed/manifest均保留；策略更正作为新收据，不覆盖旧事实。

平台原有timer可能独立执行，本代理没有平台UI控制能力，不能宣称其已取消。
不以再次连接成功推断之前未关机，也不以连接关闭证明当前已关机。

## 最新002设计的实际阻塞

前一新teacher的G1跑满500步、最高reward3且success=false；安全计数虽为0，
仍不是合格来源。最新002要求来源不合格停止采集器阶段，不能跳到Q1/Q2或训练。
本批工程检查通过没有解除该科学前置条件。Exp1仍缺有效base Gate3，base Gate4
保持not_measured；Exp2/3仍缺合格S/R与正式cohort。保留现场并报告阻塞，
不把旧pending、工程通过或用户授权误作科学准入通过。
