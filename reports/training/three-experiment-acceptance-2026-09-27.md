# 三实验交付验收：未通过

验收身份：`three-experiment-acceptance-20260927-001`。
范围是用户最终明确要求的原 Exp1/2/3 实现、真实数据、实际执行、结果回传及整轮关机；不以较早完成的 teacher 诊断代替三实验。

## 真实完成度

三个子代理已交付本地实现并结束工作，但结束状态不代表实验完成。原交付机器记录 `three-experiment-implementation-status-20260927-001.json` 明确：新增 base rollout、采集尝试、模型 forward、smoke 更新和主更新均为0，native runtime tests 未执行，未发起关机，平台电源未核验。

本次只读复连用户指定 SSH 端点仍在远端命令执行前返回 Connection closed，不能取得当前 GPU/进程状态，也不能据此确认平台停止计费。本次没有部署、启动训练或关机。

## 已确认实现缺陷

### P1：正式训练计划违反父合同的 episode 顺序

`scripts/prepare_paired_training_bundle.py:51` 将 episode 集合排序写入正式 training。
`scripts/run_smolvla_v2.py:121` 要求列表精确等于父实验 train_episodes。
父合同 `configs/vla/smolvla_450m_aloha_insertion.yaml:39` 的40集顺序以49、4、23、43开头，并非排序。
因此合法完整D inventory生成的A/B/C正式计划都会在训练前报错，不能产生更新。
应保留父合同顺序并独立验证库存覆盖；不能放宽已有父合同校验来消除错误。
现有fake loader测试未覆盖生成器到正式launcher的合同验证。

### P1：独立看门狗没有就绪握手与存活监督

`scripts/run_round_experiment_job.py:159` 只在spawn后即时读取PID身份，随后便启动worker；没有等待guard完成初始化。
`worker`（91–138行）也不检查guard身份或存活状态。因此guard启动后异常退出或被终止时，worker仍继续工作，不具备所声明的独立截止保护。
125/128行资源异常只terminate直接子进程；finally只写退出记录，没有回收剩余子进程组。直接子进程忽略TERM或有后代进程时，资源异常后的计算可能继续，直到仍存活的guard达到总截止。
作者复核确认不存在另一个外层实现补足这些行为。需要可验证的armed握手、持续身份检查和完整的有界进程组清理。

### P2：base测量遗漏推理后和终局资源检查

`scripts/diagnose_smolvla_base_five_scene.py:220` 仅在predict前检查预算；222行predict后直接进入236行step。
若最后一次推理越过工作截止或资源上限，该动作仍会执行，并可在267行返回回合结果而不检查越界。
需要在推理后、step前和终局检查，并明确CUDA峰值监控。建议2100秒外层预算不能替代注册的1800秒内部工作限制。

## 尚未交付的执行闭环

- Q2计划仍为 `protected_scene_inventory_complete=false` 且保护状态指纹清单为空；collector372行因此将正式N限制为0。保护库存构建与核验尚未补齐，当前不能给B/C提供真实准入数据。不能仅把标志改为true。
- `scripts/evaluate_paired_policy_v2.py:133` 明确输出 `geometric_recovery_event=not_measured`，原恢复指标尚未完整实现。
- 当前交付是分项入口；没有实际封存的整批执行计划、已部署队列或经回传校验触发的全轮关机闭环。新round runner主动禁止关机，这是局部安全边界，但不等于用户要求的自动收尾已完成。
- 没有native/Accelerate/CUDA测试、真实带载恢复、B/C真实batch、benchmark、smoke/reload、5000步训练和策略测量证据。

## 本次已独立验证

- 15份Python源码只读AST解析通过；未导入模型、数据、torch或仿真器。
- base 002计划的8项源码SHA与当前文件一致。
- 训练交付status中的7项源码SHA与当前文件一致。
- 三个代理交叉只读review，主代理复核上述P1对应调用点；没有为验收修改生产实现或历史证据。

检查脚本：`.work/accept-three-experiment-20260927-001.sh`。
上述检查只能确认语法、文件身份与静态缺陷，不能替代运行验收。结论为实现交付未达可执行验收，三项实验与关机验收均未完成。
