# Hestia checkpoint取回与磁盘清理完成

用户本次明确提供SSH入口并授权清理，随后要求“取完东西关机，继续在本地
排查，有需求找我”。已使用无卡/CPU模式执行，没有远端模型计算或训练。

40个native pretrained文件、共4,805,448,844 bytes已存入
`artifacts/hestia-native-checkpoints-20260911-001/`，包含320/640/960/1280四个
既有保存点。本地固定离线Linux容器检查白名单、路径、类型和大小，逐文件SHA
全部匹配历史quarter清单。未取optimizer/rng/scheduler，不能称为完整恢复备份。
文件身份验证不等于新的模型独立reload；后者继续在本地执行。

## 清理结果

只删除两个完全重复的归档；以下路径相对已登记durable root下的`rosetta/`：

| 删除对象 | bytes | 保留内容证明 |
|---|---:|---|
| `artifact_backups/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/m2-smolvla450m-way-b64-default-step0316-002-export001.tar` | 1,201,387,520 | 全部14文件成员与`artifacts/`中的对应Way导出逐SHA相同 |
| `transfers/prometheus-evidence-20260813.tar.gz` | 762,012,890 | 全部文件成员与对应`artifacts/`及`runs/`落地文件逐SHA相同 |

删除前再次复核归档自身及全部保留成员SHA、文件类型、根目录边界；先保存
收据，再unlink指定单文件。合计删除1,963,400,410逻辑bytes。数据盘可用空间
从4,488,654,848增至6,452,031,488 bytes，占用约91.64%→87.98%；块取整与
少量新控制记录使净释放量略小于逻辑文件总和。

不同训练阶段的checkpoint全部保留。特别是
`control/coverage40-unattended-20260910-001/incoming/000256`中的模型、optimizer、
rng及部分记录与正式B不同，完整保留，没有按“incoming”名称误删。
已存在的硬链接也未重复计作可释放空间。原始数据、唯一实验记录和环境未清理。

第一次Way清理在JSON布尔值解析处退出，尚未访问删除步骤；失败日志保留。
修正显式JSON解码后重新通过所有检查才执行删除，没有绕过保护。

## 关机和后续

15分钟窗口从13:21:35 UTC登记，另有独立截止守护；13:36:22 UTC在窗口内
提前调用平台关机包装器，之后本地解包校验独立完成。原包装器SHA为
`0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027`，未修改。
新的CPU控制脚本明确验证两个GPU查询返回6和`No devices were found`、没有
GPU设备节点，并保留包装器SHA、Trash不存在、无其他tmux/项目worker保护。
原有GPU模式helper未修改，实例未释放。

执行关机后SSH被远端关闭；唯一一次后续连接检查退出255。未通过控制台独立
核验电源/计费，不以SSH断线单独证明计费状态。接下来仅在本地进行模型reload
及保存点曲线诊断，保持原Hestia固定1280步的负结果，不自动重训或打开hidden。

完整证据位于`runs/hestia-checkpoint-retrieval-control-20260911-001/`，同名JSON
绑定传输、清理和关机脚本/结果SHA。没有推送。
