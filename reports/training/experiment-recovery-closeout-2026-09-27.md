# 实验接续、证据恢复与关机（2026-09-27）

身份：`qualification-closeout-20260927-003`。用户授权按最新交接/设计执行并部署看门狗与定时关机。本轮完成上一轮原始证据恢复及费用收尾；三项模型实验仍受技术前置条件阻塞，未启动。旧报告中的授权快照不是本轮阻塞原因。

## 本轮实测

用户提供的20497端口实例在08:14:26 UTC实测为RTX4090D、24564MiB，GPU利用率0%，无实验worker；数据盘可用11,921,727,488 bytes。没有重跑已完成Q任务、模型forward、optimizer、仿真、下载、提交或推送。

已将`runs/aloha-qualification-20260927-002`的完整证据恢复到本地相对路径`reports/training/aloha-qualification-recovered-20260927-003/`，保留原0字节失败archive。

- 原封存14文件、35,773 bytes逐文件SHA、清单SHA、数量与总字节数均独立核对通过。
- 原清单SHA：`570c094a1badbc295abab227312cfd04d1590fa25eb38ab110f26b8a2464ab97`。
- 新回传archive SHA：`e642e2a75f9edb13b628c292e3168c8d5149cebcb392166058b34c37e94d597c`。
- 原始`regression.log`确认29项检查通过；inventory退出0。
- 原worker报告21源文件/162,343,250 bytes校验匹配，7个row groups全部跳过，raw frames=0，hidden_loaded=false，training_ready=false。本轮验证的是该原始结果的完整性，没有重新执行源数据审计。
- 另取回原`shutdown-request.json`，与当前远端重新计算SHA一致。它在原manifest封存之后产生，因此单列于`remote-extra.sha256`，不冒充原14文件的一部分。

## 本轮独立关机保护

先设置平台定时关机并刷新确认`2026-09-27 16:22:00 Asia/Shanghai`；新namespace为`runs/qualification-closeout-20260927-003`。

复用此前content-addressed workspace中的guard，并核对其SHA与本地一致：`53bc4c8af8cfb8107cc8929314668fb2b149b46cb46e08ca50f4d6924531baaa`。没有覆盖旧代码或旧run。新控制登记一次性传入，guard以独立session启动，PID1543，工作截止08:20:33 UTC，关机截止08:21:33 UTC；无模型worker，无自动retry/resume。

取回新`registration.json`与`guard-armed.json`并核对远端/本地SHA之后，写入create-only `closeout-now`，让guard提前执行受保护关机。未通过UI执行立即关机。08:20:05 UTC附近刷新平台，目标实例明确显示**已关机**，早于平台16:22兜底；见同接收目录`platform-stopped.png`及`platform-closeout.json`。这确认平台当前停止状态；本轮新guard的最终shutdown-request未再回传，不能将计费停止的精确秒数或平台内部因果归属作为已测事实。

未释放实例、删除数据或取消兜底定时设置。

## 实验仍缺什么

1. Exp1只补base Gate4，保持其他原样。最新交接/准入报告仍缺同artifact/code/simulation-plan绑定的现成有效base Gate3；已存历史base Gate3失败（9次关节限位、1次意外接触），不能据此启动Gate4。没有重跑/修改Gate3，也没有重跑trained端点。base Gate4保持not_measured。
2. Exp2/3没有已确认的同域S/R matched cohort、N、完整sampler、恢复bank及新runtime预算。回传Q证据确认当前reader因混合protected row groups而未读样本；缺少视频/真实成功与状态条件恢复标签及protected scene/parent清单。文件下载/hash通过不等于恢复示范合格，跨机器人源不能填入ALOHA训练。

后续先在本地设计并验证隔离reader，确保state/action/image进入审计之前完成protected样本隔离；再以新身份执行有范围的数据资格检查。即使reader修复，真实成功/恢复标签仍须可复核证据，不能由next.done或轨迹名生成。缺有效base Gate3时，修复/重跑它属于原Exp1以外的设计变更，不能悄悄加入原比较。

本轮状态：证据恢复与平台关机完成；三项科学实验阻塞；M2不变。阻塞不是缺少本轮SSH授权。
