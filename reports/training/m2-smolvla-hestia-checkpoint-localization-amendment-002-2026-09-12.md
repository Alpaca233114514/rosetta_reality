# Hestia 保存数组定位：002 形状适配修正

001 在首次数组校验处报 `ValueError: Mask shape differs`，尚未计算误差分解。
新诊断器错误地将 `max_action_dim=32` 当作保存 prediction/target/mask 的宽度。
现有 collector 与只读数组检查确认：两种空间的 prediction 均为 `[4,45,50,14]`，
target/mask 为 `[45,50,14]`；只有 noise 为 `[4,1,50,32]`。这是诊断器缺陷，
不是模型或原始证据不一致。

修正仅区分合同中有效动作宽度与原生 padded noise 宽度；仍要求全部有效槽、
严格布尔 mask、有限值与精确历史数组匹配。新增反例覆盖有效/噪声宽度不同、
错误 padded mask、无效槽和错误 prediction 宽度。七项反例须通过才执行 002。
比较点、场景、窗口、夹爪阈值、分解公式、容差与资源预算均继承原计划。

001 的计划、当时脚本/测试副本、执行日志与 `failure.json` 原样保留，分别位于
`runs/hestia-checkpoint-localization-preflight-20260912-001/` 和
`runs/hestia-checkpoint-localization-20260912-001/`。002 使用独立计划 JSON 与
新目录 `runs/hestia-checkpoint-localization-20260912-002/`，不覆盖或续写 001。
用户本轮本地诊断范围内继续修复；没有训练、模型推理、远端操作或门槛放宽。
