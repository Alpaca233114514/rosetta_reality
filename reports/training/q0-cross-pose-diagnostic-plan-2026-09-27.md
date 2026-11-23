# Q0 independent cross-pose diagnostic

用户最新授权取消任务成功类执行依赖：前一场景失败、refusal 或物理安全中止后，独立下一场景仍执行。此变更仅适用于新 diagnostic，不修改冻结 Gate、历史结果、teacher 参数或既有验收阈值。G1 成功不是此入口的前置条件。

入口 `scripts/diagnose_teacher_cross_pose.py --output {job}/cross-pose --max-seconds 600`，外层 supervisor 负责硬超时。固定使用 `coupled_translation_teacher`、原 Action Contract、observer 与环境；只执行未运行的 seed 1900–1904，各至多 500 步；不重跑 seed 10。零模型调用、数据加载与 optimizer 步骤。

逐步 trace 记录原/裁剪动作、机器人状态、物体/EEF 位姿、phase、reward、contacts 与物理安全计数。原合同 clipping 保持并记录；nonfinite 动作不得进入 `step`。物理 limit/collision 立即结束当前 episode，然后继续下一 seed。NaN、内部异常停止整个任务；deadline 到达停止后续执行。每 seed JSON 和汇总 create-only；完整五 seed 分母保留，未测 success 为 null，并独立报告 successes、measured_attempts、registered_attempts=5。

成功只能取原 `info.is_success`，reward 3 不算成功。此 diagnostic 比原 G2 更早停止不安全 episode，因此不能称为原 G2 完整同协议结果。所有输出固定 `provider_qualified=false`、`formal_gate_result=null`；任务失败不阻断下一场景并不等于 provider 通过验收。此文件只登记新入口；真实执行证据尚未产生。
