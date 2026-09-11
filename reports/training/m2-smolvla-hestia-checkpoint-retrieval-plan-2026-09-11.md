# Hestia现成checkpoint取回计划：不重训

状态：本地准备完成，尚未取得本次传输/开机授权，未连接远端。当前实例由用户
确认已关机。此计划不授权重开计费GPU，也不沿用已经结束的Hestia训练预算。

现有预测已能确认C训练拟合强、开发左臂映射差；二维覆盖、坐标读出和时移
对照不能修复完整问题。下一项有区分力的模型检查是固定320/640/960/1280
四个保存点的同协议离线曲线：失败是否随拟合增强而出现，还是从最早保存点
就存在。不能用B的256步代替C的320步，因为其学习率轨迹不同。

本地已再次查找`artifacts/`、`checkpoints/`、`runs/`：只有恢复的预测bundle，
没有Hestia权重。四个模型已在远端存在；无需启动新的训练。取回以后再用本地
固定Linux容器逐个分析，禁止同时驻留四个模型，保持16GB本机预算。

## 具体文件和执行边界

来源相对已登记durable root：
`checkpoints/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/smoke/m2-smolvla450m-visual-hestia-fit40-001/checkpoints/`。

仅取`000320/000640/000960/001280`各自`pretrained_model/`下已登记10文件，
包括模型、config、前后processor及normalization、tokenizer与train config。
精确40文件、4,805,448,844 bytes（约4.48GiB）；每份模型1,197,789,256 bytes。
同名JSON逐文件固定来源相对路径、目标相对路径、大小及历史SHA，已核对恢复的
`C-integrity.json`与本次关机前清点记录；启动后仍必须重新核验，旧清点不是
当前远端状态证明。

本地目标为新目录`artifacts/hestia-native-checkpoints-20260911-001/`，create-only，
先检查空间至少12GiB。中断文件独立保存，不覆盖历史内容。逐文件SHA与历史
quarter清单严格相等后才接受；大小、路径、文件类型或哈希不一致立即停止。
拒绝symlink、路径逃逸和白名单外文件。不会复制optimizer/rng/scheduler，
因此这不是完整训练恢复备份，不能据此删除远端任一checkpoint。

授权请求范围是无卡/CPU模式下的一次只读清点和传输；服务器只读文件及计算
checksum，不import模型、不运行GPU、训练、评估、更新源码或安装依赖。
若无卡方式不可用，停止并报告，不自行改为计费GPU模式。

远端窗口上限900秒，结束或超时后停止本次传输，保留部分文件，在120秒收尾
窗口调用已登记关机helper。关机前复核helper摘要、没有其他任务、保护条件
成立；不终止别的任务、不删除Trash、不释放实例。SSH断线不等于电源/计费
确认，关机后应由平台或用户确认。具体SSH身份以本次用户提供信息及known-host
核验为准，不在仓库保存主机地址或凭证。

## 取回后的本地科学边界

先做完整身份/参数/processor检查和实际1280 artifact本地独立reload，再冻结
跨设备数值复现与四保存点评估协议，不能在看到新结果后放宽容差。XPU不支持、
非finite、资源超限或历史C复现失败即停，不转移到远端GPU。

四保存点评估只回答拟合过程与开发映射变化，原Hestia固定末步1280的负结果
保留。不得把最好的开发保存点事后改写为原实验已通过，也不打开hidden或把
离线checkpoint比较当作闭环成功。新训练轴必须另行登记和授权。
