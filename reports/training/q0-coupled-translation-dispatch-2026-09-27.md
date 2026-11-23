# Q0 候选一次性部署记录

执行身份：`q0-coupled-translation-20260927-001`。
本记录是启动证据，不是 teacher 资格、采集或训练成功报告。

36个文件经新内容寻址工作区上传并逐文件SHA核验通过。
工作区相对数据根：`workspaces/20260927T110136Z-66a5bbcdbdcd-cbc3016cb3b9`。
包SHA：`30a7ac4700ffdb2ece4f67ae68442f99b494129605622c7d96ed48ca311fa1ca`。
组合身份：`cbc3016cb3b9a98bfa19ef23bef8ef445b56e98aae57a451c6872c6534fbe746`。
计划SHA：`57a2e1b9f345a200a4ed3e825a63b4f015c5cd06017dbd2fa2bf5e10eb236b45`。

部署前在同一AutoDL容器完成10项标准库回归，0.206秒、全部通过。
包括组长退出后清除忽略TERM子进程、无法证实归属时拒绝关机、
G2成功不掩盖安全失败、重复attempt拒绝、源路径/时限/关机授权边界。
候选synthetic、benchmark及teacher gates由后台worker顺序运行，未在此提前认定通过。

启动于2026-09-27 11:03:20 UTC，worker PID1966，独立guard PID1965。
11:03:59 UTC一次启动核验：两个进程都活着、各自独立进程组且父进程为1；
原SSH会话退出不影响运行。guard armed收据与计划截止一致。
worker已记录运行环境并进入check_env阶段；该时刻尚无该阶段完成收据。

工作截止：2026-09-27 19:48:20 +08:00。
看门狗关机截止：2026-09-27 19:53:20 +08:00。
正常完成或失败后会提前进入最多300秒回传缓冲，再请求受保护关机。
平台20:00规划截止由用户接管设置，未由本代理核验；运行中不延长预算。
无自动retry、resume、训练或额外采集。没有删除数据、提交或推送。

结果远端相对路径：`runs/q0-coupled-translation-20260927-001/`。
一次有界本地接收器已在启动前就位，等待worker完成后取回新归档并核对整包SHA；
接收器依赖本地连接存续，服务器durable输出独立保留。该启动时刻回传尚未完成。

入口及边界见 `q0-coupled-translation-execution-plan-2026-09-27.md`。
实际阶段结果以新run的phase JSON、benchmark、failure/result、manifest和worker-exited为准。
G0–G2全通过也仅为partial_screen_only；完整Q0的G3/G4实现与G5独立审计仍缺。
当前 `provider_qualified=false`、三臂训练未启动，base Gate4维持not_measured，M2不变。
