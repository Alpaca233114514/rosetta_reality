# Teacher 独立场景诊断结果

身份：`teacher-cross-pose-diagnostic-20260927-002`。按照用户最新要求，取消独立场景之间的任务成功前置依赖，保留每回合物理异常中止、身份核验和资源上限。本次测量完成；整体三实验未完成。

## 结果

同一候选此前 seed10 的 G1 为500步、最高reward3、success=false。本次未重跑seed10，独立运行全部五个已登记场景。

| seed | 环境步 | 最高reward | 结果 |
| --- | ---: | ---: | --- |
| 1900 | 289 | 4 | success=true |
| 1901 | 500 | 3 | task_failed |
| 1902 | 40 | 0 | safety_stop；1次意外接触计数 |
| 1903 | 500 | 3 | task_failed |
| 1904 | 500 | 3 | task_failed |

共1829环境步，全部5个场景均有测量，成功1/5；耗时96.251秒。所有场景关节越界与action clipping计数均为0，无refusal。1902的物理异常只中止该回合，1903和1904继续执行。

因此，seed10失败不能推断所有后续场景失败。该观察支持独立测量后逐项报告，而非按任务成功串行阻断。它不证明全部标准通过、来源已合格或泛化已修复。由于诊断在物理异常时提前结束回合，`original_g2_protocol_equivalent=false`，不能将1/5改写为原正式G2结果。`formal_gate_result=null`、`provider_qualified=false`，模型forward和optimizer更新均为0。

## 复核与身份

- 部署前和worker内分别执行相同20项测试、12个subtests，全部通过；重复运行不合并为额外独立测试。
- 归档回传整包SHA通过；manifest绑定20文件、7,214,979 bytes，本地逐文件SHA及大小全匹配，worker-exited对manifest的绑定匹配。
- 本地独立重读全部JSONL，逐seed重算步数、success、最高reward、关节越界及意外接触计数，与摘要一致。
- plan SHA：`5e72c8d8e07728b3395cb69451402fa04304da98cef09ec8b837310910e321ca`。
- workspace composite SHA：`67b68d8c4105f37241d62792d77b42d6fb0af80d9716e5799a6ee45023925209`。
- manifest SHA：`bc90dd853da59fd559a82c95d292a299ff75f213e394a75e1dfb0b3f23a41e11`。
- summary SHA：`35cc9637d2d12c75e3a7d590c7885df3c0dd437c72a83fe6ae8b4b820201c6fa`。
- 原证据：`.work/teacher-cross-pose-diagnostic-20260927-002-received/extracted/`。

## 生命周期与未完成部分

前一身份001因遗留未确认的20:00平台时间窗在准备阶段退出，未运行场景；其失败证据已回传并核对，保留不覆盖。002使用null平台时间（用户管理timer，未确认具体设置），仍有780秒独立工作看门狗、600秒诊断上限。没有延长运行中的注册预算。

002 worker于19:54:07启动，19:55:50退出。看门狗19:55:51确认worker进程组为空后退出；全部完成/失败路径无平台关机调用，收据shutdown_requested=false。不得把此批完成当成整体三实验完成，也不以这些收据证明平台实时计费状态。

原正式模型Gate3/4保持历史事实。Exp1的base Gate3此前失败涉及限位和意外接触，并非单纯任务成功率不足；base后续可另行登记有安全中止的独立任务测量，不能伪造passed Gate3。其独立入口与当前远端权重/cache仍待核验。Exp2/3仍缺真实配对S/R、事件标签及正式训练输入，取消成功前置条件本身不会生成这些数据。
