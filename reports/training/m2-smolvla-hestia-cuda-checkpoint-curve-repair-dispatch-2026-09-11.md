# Hestia曲线002：部署修复已通过现场原生检查

用户重新提供SSH授权补救。执行版本`0308e8d7ed18e8f8cd44aed051a7e0baef4b5d8b`，
使用新的独立workspace与`runs/hestia-cuda-checkpoint-curve-20260911-002`。
001失败记录未改写。未训练、未下载；后续推理在本次固定预算内自动完成或失败停止。

本地22项检查通过，包括真实native normalization的缺失入口复现、绑定后完整通过、
重复绑定与SHA漂移拒绝。现场完整native normalization检查通过：report SHA
`263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988`，view manifest SHA
`9853c191ae87016379fc1a16ebfbb87e05ab5147cd8a03d82f9c2a894c9b531e`，9个view文件核验。
此检查权重加载前执行，随后现场20项CPU保护检查通过。

1280保存点首个固定控制输出相对原CUDA normalized数组逐值相同：
`max_abs=0.0, mean_abs=0.0, atol=0.0, rtol=0.0`。
这证明本次部署已越过原缺失入口并通过首个控制，不代表完整180条件或四保存点曲线完成。
最终曲线、参数前后SHA及其余保存点结果仍以002闭合结果为准，Gate3/4及M2状态未改变。

2026-09-11 15:11:45.682 UTC启动supervisor PID1539与独立watchdog PID1546，均已核验
启动tick并脱离SSH。最多480秒预检/推理，600秒硬截止为15:21:45.682 UTC。
本地隐藏接收端PID24504等待封闭manifest，逐文件SHA核验后回写关机收据；没有收据也
由远端硬截止触发受保护关机，不释放实例。当前启动记录不证明最终回传或实际关机。

证据：`runs/hestia-cuda-curve-preflight-20260911-002/dispatch.log`与`health.json`。
首个只读健康检查因本地命令引号错误失败，改为文件标准输入后成功读取上述状态；
没有因此重启或修改运行中的任务。结果接收目录为
`runs/hestia-cuda-curve-receiver-20260911-002/`，核验后落入
`runs/hestia-cuda-curve-recovered-20260911-002/`。
