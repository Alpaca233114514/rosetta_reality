# 三臂固定终点测量回传记录（2026-09-30）

最新已找到的运行身份为 `paired-ABC-measurement-20260928-003`，对应 2026-09-28 启动的 A/B/C 固定 5000 步终点测量。本次仅只读取回与检查保存结果，没有启动训练、模型推理或仿真。

## 回传完整性

接收根：`.work/paired-ABC-measurement-20260928-003-received-20260930-001/`。远端运行目录的全部普通文件共 **151/151** 逐文件 SHA256 校验通过，解包文件在 `files/`；`remote.sha256`、`verification.log`、`archive-members.txt` 和 `results.tar.gz` 保留在接收根。三个指向既有模型目录的符号链接单独记录在 `remote-symlinks.txt`，此次没有沿链接重复下载 checkpoint。

- 原始压缩包 SHA256：`2e960ece6dcd492400e5bd31d69f977286d0a807cfd144f5bd5be3ab4284fbb0`。
- 远端文件清单 SHA256：`efcc8cb4ae0d5dd509b57be8a87933afbb1dfc51d12b8fa84673aa6c85b7bfc3`。

## 保存结果

`files/worker-exited.json` 的九个阶段退出码均为 0，`error_type=null`，清理状态为 `owned_group_empty`。各臂 `summary.json` 为 `status=completed`、五个场景全部测量、`optimizer_updates=0`。以下数值直接整理自回传汇总，尚未独立重算完整轨迹或重跑评价。

| 臂 | 五场景任务成功 | 五场景物理安全停止 | 恢复 bank 任务成功 | 几何恢复 / 已验证恢复链 | 两独立进程完整动作数组重载 |
| --- | ---: | ---: | ---: | ---: | --- |
| A：D | 0/5 | 3/5 | 0/8 | 0/8、0/8 | passed，精确一致 |
| B：D+R | 0/5 | 4/5 | 0/8 | 0/8、0/8 | passed，精确一致 |
| C：D+S | 0/5 | 4/5 | 0/8 | 0/8、0/8 | passed，精确一致 |

恢复 bank 预登记 10 个状态，每臂实际测量 8 个；2442、2448 的分支未生成，记录为未测量，不计为失败。其余 8 个均以物理安全停止结束。五场景中 A 记录 3 次意外碰撞、0 次关节限位违规；B 为 1 次意外碰撞、3 次关节限位违规；C 为 0 次意外碰撞、4 次关节限位违规，并记录 1 个原始动作裁剪字段。

各臂的主要证据为 `files/measurement-{A,B,C}/summary.json`、`files/measurement-{A,B,C}/recovery-bank/bank-summary.json`、对应 JSONL 轨迹，以及 `files/reload-proof-{A,B,C}.json` 和两份 probe 文件。原始计划、注册记录、doctor 与阶段日志一并取回。

## 解释边界

保存结果未显示 B/C 在本轮任务成功或恢复成功上优于 A。三臂重载一致不等于任务成功。汇总明确标记 `original_gate4_protocol_equivalent=false`、`formal_gate_result=null`；本轮不能记作正式 Gate 4 或 M2 验收通过，也不能由这一探索性比较推断恢复数据普遍无效或定位唯一失败原因。

退出收据记录 `shutdown_requested=false`；本次未操作平台关机、释放或删除，未核验平台当前计费状态。原始远端文件与已有本地修改均保留。
