# Hestia 参数互逆首次调度：入口类型错误，未运行模型

用户在带卡执行门前提供 SSH，实测原 RTX 4090 D 空闲，54 个已登记身份
复核通过；复用了无卡阶段准备的唯一工作区，没有重新上传。001 supervisor
与独立 watchdog 均已启动且 PID/start_ticks 匹配。

worker 在 27.966 秒退出，`completed_conditions=[]`，错误为
`Condition base1280 collector exit 1`。只读状态快照看到 doctor、normalization、
CPU 检查 exit code 均为 0，随后 collector 在模型加载前的 profile 哈希检查报：

```text
AttributeError: 'str' object has no attribute 'open'
```

原因是 `file_hash` 要求 `Path`，入口却把环境变量 profile 字符串直接传入；
模板文件名迭代也有同一错误。此前 34 项合成检查没有覆盖实际 CLI 的完整
前置链。该失败属于本次诊断实现，不是 Hestia 模型的新负结果或视觉泛化
根因；四组推理均未开始，没有 checkpoint 选择、optimizer 或新 Gate。

只读观察时 worker 已退出、子进程不活跃、handoff manifest 已准备好。
接收尝试记录在 `runs/hestia-parameter-crossover-receiver-20260912-001/`：
archive 为 0 bytes，stream.log 记录远端关闭连接。其时间在 worker 退出后
120 秒回执等待结束约 5 秒之后。随后再次只读连接仍被关闭。
完整失败黑匣子及实际关机 wrapper 记录尚未取回，不能声称已完成回传或
独立确认平台电源/计费。旧远端结果、原源码/模板和本地失败接收目录保留。

修正版以新 run `hestia-parameter-crossover-20260912-002` 登记。原 001
collector 文件没有修改；新 `_v2` 文件的实质差异仅两处 `Path(...)` 转换，
另有新 run ID。原 SHA、精确复现容差、卡型、数据、动作合同和实验轴不变。
详见同日期 `parameter-crossover-v2-plan` MD 与 `parameter-crossover-v2-template` JSON。

43 项本地测试通过（9.73 秒）：包括 34 项既有检查和 9 项 CLI 回归。
CLI 回归真实调用参数解析、PID/许可、文件 SHA、checkpoint 身份和设备门禁，
使用合成文件并替代最终 collect/CUDA 可用性。它复现旧版异常、验证新版
四项分派，并保留输入篡改和无卡拒绝；不是实际 CUDA 成功证据。

002 的六文件修复包已准备，71,680 bytes，SHA
`a7bafef3e5a30f4c41b1fe41719c3c72e64633bfc39d42c73bcfc97af6aadd5e`。
只含新代码、测试和计划，不含新权重/原始数据。59 个科学身份绑定已冻结。
增量 stager 支持从既有组合身份派生新工作区；旧 stager 已另存并核对原
SHA，旧工作区只读。新增文件不会覆盖原 001 文件或复制其 normalization 绑定。

本机新 `run.sh` 编排先确认结果接收器已连接，再启动 supervisor，避免把
回传工具申请留到 worker 退出后的短等待期。新接收目录与旧失败目录分开。
八份本机 Bash 文件语法通过；新 stager/接收编排尚未在远端实跑，不作成功声明。

下一步需要可用 SSH：先恢复 001 失败证据，再部署修正版；无卡可完成这两步
及 CPU 检查，四项推理仍须原 4090 D。当前尚无新的模型推理结果，根因目标
未完成，M2 未完成。
