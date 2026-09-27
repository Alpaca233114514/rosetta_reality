# 三项实验执行准入核验（2026-09-27）

身份：`execution-readiness-20260927-001`。本轮用户明确授权按最新交接和设计执行实验、一次性上传并部署看门狗与定时关机。历史交接里的禁止执行状态属于历史授权快照，本轮授权已更新；本次阻塞原因是技术前置证据缺失。

## 实际执行与结果

已读取 `experiment-handoff-2026-09-27.md`、`three-experiment-design-2026-09-27.md`、配套设计 JSON、架构入口与 Faust/Zen 审计。本轮连接用户指定的原实例，在 AutoDL 平台容器内执行有限的报告/收据元数据核验，没有模型加载、数据行/视频解码、forward、optimizer update 或仿真 rollout。

远端实测 GPU 为 RTX 4090 D、24564 MiB；cgroup 内存限额 64,424,509,440 bytes；准入核验时数据盘可用 11,955,863,552 bytes。此为资源观测，不是训练 benchmark、doctor 或训练存储预算通过证明。

一次性上传新身份的核验脚本、独立 guard、原设计/交接及 runtime 参考。归档及 8 项 payload 校验通过。新输出位于远端相对 namespace `runs/execution-readiness-20260927-001`，原证据和已有修改保留。

| 实验 | 已确认依据 | 当前状态 |
| --- | --- | --- |
| Exp1，仅补 base Gate4 | 远端 `runs` 内有限报告扫描找到6份 Gate3 元数据，未找到对应 prepost base/固定base权重的候选报告；本地已取回的 `gate3-smolvla-sim-481.json` 为 base，明确失败，关节限位9、意外接触1 | 缺现成有效 base Gate3 同源绑定；Gate4 `not_measured`，不补0/5，不重跑训练后端点 |
| Exp2/3 | 当前设计 `N=null`，合格S/R清单、完整sampler与恢复bank尚未登记；跨机器人恢复源不符合当前ALOHA合同 | 未进入数据资格、准入smoke或训练；不是证明所有候选数据都不可用 |
| 下载完整性 | 远端现成最终收据与交接一致，212文件、66,158,775,113 bytes，`complete=true`、无失败；收据hash已保存 | 本次核对收据，不是重新读取全部66 GB，未验证真实行、视频或恢复标签 |

远端 Gate3 报告扫描不跟随symlink，只扫描 `runs`；不包含所有 workspace、artifact payload或另一个已关机clone。本次未把报告中缺少直接base引用推定为全面排除所有可能的base证据；其现成有效绑定仍未找到。未执行完整Q资格审计，也未声称不存在可用恢复数据。

## 截止、失败与关机

核验任务工作上限180秒，独立guard硬关机截止900秒；worker完成后预留120秒回传窗口，均从登记时刻开始，不依赖SSH连接。只读元数据扫描耗时约0.083秒，无自动训练、retry、resume或扩容。

guard已启动并保存armed记录，但在检查已退出worker时漏捕获`FileNotFoundError`，提前关机分支退出。此为本轮控制脚本缺陷，失败日志保留在 `runs/execution-readiness-received-20260927-001/guard-failure.log`；不能声称该guard成功实现关机。

平台另外设置并在实例行核验了北京时间 `2026-09-27 13:45:00` 定时关机，随后直接通过平台请求立即关机。刷新后的同一实例行明确显示“已关机”，见 `reports/training/experiment-execution-platform-closeout-2026-09-27.json`；截图作为本聊天附件保存。未请求释放实例或删除数据。

## 回传证据与接续

本地新接收目录：`runs/execution-readiness-received-20260927-001/`。14个文件的远端声明SHA与本地重新计算一致，原始archive SHA为 `05414b06d9ec07fa3e9c8ab38623ded941ade01c0b9f7af28670a2780fbe6d6e`。该完整性检查不增加独立模型实验，也不证明真实数据标签资格。

执行目标仍未完成。Exp1需要提供或确认同artifact/code/simulation-plan的现成有效base Gate3；设计明确禁止在此待办内新增Gate3修复/重跑或降低门槛。Exp2/3下一技术阶段为同域ALOHA的Q资格审计，然后才可封存N、匹配清单与训练/评价身份，并依据实测benchmark登记正式工作时长、存储与关机预算。恢复bank、teacher/人工标签与采集适配不能通过填写虚假通过状态代替。
