# Hestia CUDA保存点曲线：部署前置文件缺失，未测得曲线

用户在本轮提供GPU SSH入口，授权一次性部署只推理任务、看门狗、结果回传和自动关机。
执行版本`f17366df933c9ef534001f7041f7be6ae468a625`，新建独立detached workspace。
既有训练、checkpoint与历史证据保留，没有重训、下载、选择保存点或自动重跑。

## 实际结果

- 2026-09-11 14:51:58 UTC登记supervisor PID1530、watchdog PID1537；均以启动tick
  核验身份，脱离SSH，硬截止15:01:58 UTC。
- 现场doctor通过：RTX4090D、PyTorch2.8.0+cu128、LeRobot0.6.2，原模型/数据/cache
  manifest及配置身份通过，源码工作区clean；现场20项CPU检查通过。
- 11.919秒后首个1280 collector退出1，错误为
  `ValueError: Version-2 normalization report checksum changed or is missing.`
- 错误发生在native `_validate_normalization`，在policy权重加载与首次forward之前。
  完成保存点为空，optimizer_steps=0。1280控制及320/640/960曲线均为`not measured`，
  不构成同设备数值失败、训练结论或Gate3/4/M2进展。
- 本地隐藏后台接收端已自动回收7个结果/日志文件，全部长度和SHA匹配；manifest SHA为
  `f7d23867fb7343129880503356b7bd44193ec66bc036dd3a8d3394036306ab4d`。
  已回写关机收据。随后一次只读SSH关机检查返回连接关闭；未取得shutdown-request内容，
  未核验平台账单状态，不将连接关闭单独称为计费确认。

## 原因与后续边界

这是本次部署包遗漏：历史main1280要求仓库相对的train-only normalization report和
dataset-view manifest；native验证器先从代码workspace解析这两个入口，再验证manifest
所属目录等于durable run root下的dataset view。bootstrap只放入C数组与main1280，
仅设置`ROSETTA_RUN_ROOT`不足以满足入口。现场doctor验证全局cache身份，20项CPU保护
检查未覆盖这组workspace绑定，不能替代真实部署前置文件检查。

下次部署应在加载权重之前，核对既有durable report/view SHA并在新workspace建立满足
native目录身份断言的create-only入口；先做不加载权重的完整native normalization检查。
不得放宽校验、重写历史计划或复制manifest到错误父目录来跳过目录身份约束。
本次已经按失败停止，没有重新开机或启动下一轮。

证据目录：`runs/hestia-cuda-curve-recovered-20260911-001/`；接收收据位于
`runs/hestia-cuda-curve-receiver-20260911-001/receipt.json`；部署登记原始输出位于
`runs/hestia-cuda-curve-preflight-20260911-001/dispatch.log`。
