# Hestia 原生路径采集准备：本地已验，远端未执行

根因目标保持进行中。阶段桥接的17项检查和764个独立重算字段已完成；
单一左开合阶段不能充分解释原生 C 的朝向失败，见同日期 phase-bridge 结果。

下一阶段计划与模板为 `m2-smolvla-hestia-prefix-path-plan-2026-09-12.md`
及 `m2-smolvla-hestia-prefix-path-template-2026-09-12.json`，状态明确为
等待当前 SSH 及原4090D仅推理/受保护结束授权。没有启动外部计算。

本地已实现并核验：

- `scripts/diagnose_hestia_prefix_path.py`：只读前后hooks；仅原1280，180次
  native forward；沿用原collector的当前data/normalization/model/processor/
  非视觉恒定/原CUDA逐值控制，增加实摄像头64 token及逐噪声不变性检查。
- `scripts/run_hestia_prefix_path.py`：保留旧独立watchdog、PID身份、受保护
  shutdown和480/600秒上限，只将step列表收紧为1280并换新入口及测试。
- `scripts/stream_hestia_prefix_results.py`、`scripts/verify_hestia_prefix_results.py`
  和 `scripts/receive_hestia_prefix_results.sh`：64MiB白名单流、原子新目录
  校验、匹配receipt后允许结束；不复用旧接收失败目录。

固定离线Linux Docker中32项合成/协议测试通过，Ruff/format通过。包括完整
包装器在toy policy上的180次路径，以及合成失败后 factory、policy method
和全部hooks恢复。补充显式仓库导入路径后，相关11项重验通过，接收器
Bash语法通过。未运行真实模型、SSH、GPU或关机入口。

静态读取本地已取回1280 safetensors的73,184-byte JSON header，确认四个
被观察投影均为320×320；全文件SHA再核对为原C的`21b4dd93...`。没有
materialize权重张量或调用模型。8组×64token×320宽×45行×float32为
29,491,200 bytes，低于48MiB prefix原始数组预算。全文件SHA会读取原始
文件字节；这与解析权重张量不同，原header检查记录的命名澄清另存保留。

真实模型的hook数量、实际CUDA激活与性能、180个输出逐值一致、完整回传及
计费结束状态均仍为`not measured`。准备检查不能冒充这些现场结果。
源码基于原成功CUDA collector及守护器，旧源码未改；同名JSON绑定新源码、
计划/模板、前置日志和静态shape证据。正式传输须使用现有
`scripts/stage_autodl_from_wsl.sh`的新content-addressed workspace，先核对
现场身份再运行，不手工覆盖旧 authority。没有提交、推送或删除。
