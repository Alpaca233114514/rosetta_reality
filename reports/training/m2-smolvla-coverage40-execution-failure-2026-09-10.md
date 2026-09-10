# Coverage40 无人值守执行失败复核

2026-09-10 15:03（UTC+8）通过已授权 SSH 只读复核：同一 RTX 4090 D 可连接，
GPU 使用率 0%，显存 1 MiB，无 CUDA 计算进程。当前可连接不代表历史关机未执行，
也不能独立证明平台计费状态。

运行源码 `3adaca09b5e758a256086c9cdc6f79b9e60e3600`；远端工作区相对 durable
root 为 `dev/visual-coverage40-old4090d-20260910-3adaca0`，证据目录为其中的
`runs/visual-coverage40-unattended-001/`。源码身份与启动回执一致。

任务 14:02:53 启动，14:03:28 在 `preflight-b1` 失败，14:03:38 supervisor
保存退出码 1 和 `shutdown-request.json`。没有超时后继续、重试或启动 B。
关机 wrapper SHA 与登记相符，无 `shutdown-blocked.json`；平台关机及计费
状态仍未独立核验。

- 已通过：check-env、doctor、57 项关联回归、1 项真实缓存检查、benchmark。
- 未执行：真实 batch-1 forward、batch-4 preflight、smoke、A 复评、B 训练、
  两臂统一评估及 independent reload；这些结果均为 `not measured`。
- 失败包装器资源：GPU allocated/reserved 均 0 bytes，host RSS 624197632 bytes；
  学习率及采样记录均为空，optimizer 步数 0。未进入权重加载和 forward。

原始错误：

```text
ValueError: The version-2 AdamW optimizer contract is invalid.
```

原因已在远端对实际 plan 只读重现：`prepare` 用 JSON 保存计划，但
`load_v2_plan()` 经 `_load_yaml()` 无条件使用 `yaml.safe_load()`。
`eps: 1e-08` 和 `weight_decay: 1e-10` 经 JSON 解析为 float，经当前 YAML
解析器解析为 str；`validate_optimizer_contract()` 因类型不符正确拒绝。
这是新守护计划生成与真实入口之间的工程缺陷，不是 GPU 故障或视觉学习负结果。
原有静态测试没有覆盖生成文件经真实 loader 再校验的完整路径；之前孤立验证
不能证明这个入口可执行。

下一步先修复格式与 loader 的一致性，并增加实际写盘计划经真实入口读取、
四阶段全部 schema 验证的回归；保持优化器数值、训练与评估合同不变。保留原始
失败目录与计划，用新版本登记后再决定是否重新执行，不自动重开本任务。
本次复核未改服务器代码、加载模型、重训或操作电源；视觉泛化仍未修复，M2 未完成。
