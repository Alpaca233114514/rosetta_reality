# SmolVLA 原生视觉控制：权威文件核验 — 2026-09-10

**已找回原 8 场景控制 checkpoint 和最终源码身份；没有运行新训练或模型评估。**
此前老实例未找到文件的记录保持原样。本报告更新的是随后取得的原实例证据，
不把找回文件、历史 47 项回归或训练场景通过称为视觉泛化修复；M2 未完成。

## 核验结果

权威副本位于远端 durable root 下 `dev/visual-utilization-20260909-001`。
分支为 `codex/smolvla-visual-conditioning-repair-20260909`，HEAD 为
`14f320d981e7fdc53142a314e6d4cc9f3ea58940`，工作树有未提交修改。
本地已推送快照 `964871b4e46250061b8b4391aae8070e34aeebef` 不能代表该源码身份。
不得在这个 dirty 副本直接 `git pull`，也不得用本地文件覆盖它。

| 对象 | 本轮只读核验 |
|---|---|
| 最终审计绑定的源码/文档/报告 | 13/13 文件 SHA-256 与记录一致 |
| 旧 -003 plan 绑定实现 | 11/11 文件 SHA-256 一致；与上一行有交集 |
| 历史评估、reload、更新、资源和退出证据 | 6/6 SHA-256 一致，原预登记哈希一致 |
| parent/runtime/Gate 1/Gate 2/normalization/train-view manifest | 6/6 文件哈希一致；不等于当前完整环境预检通过 |
| step-256 权重 | 1,197,789,256 bytes；在服务器计算的哈希与旧报告一致 |
| 完整恢复状态 | optimizer、scheduler、RNG、step 文件仍在，已计算文件哈希；不用于续训 |
| normalization/unnormalization | 两个状态文件哈希一致，均为 `2021037779c22e8ab7c3344d9d8f7e5ce65a8c13326eab3201eb04eff2aff25d` |
| 保存的 eval/reload `results` | 逐值相等；仅覆盖原文件实际保存的字段 |

控制 checkpoint 的 durable-root 相对路径：

```text
checkpoints/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/smoke/m2-smolvla450m-visual-native-b4-pilot-003/checkpoints/000256
```

其中 `pretrained_model/model.safetensors` SHA-256：
`fd95a19ca8d8f9e1c6a5c477807aaf69fc3e19bb15e626bd53e4224da8ad3a5c`。
精确文件路径、大小和哈希见同名 JSON；大文件未传回本地。

九份审计绑定的关键源码按原字节存入
`reports/training/visual-native-authority-20260910/`，合计 105,691 bytes，
以 `.py.txt` 保存；`manifest.json` 记录原路径和哈希。它们是选择性恢复归档，
没有安装到当前 `src/`、`scripts/` 或 `tests/`，也不是完整远端工作树备份。
旧五份格式化前片段保留。原始报告 md/json 已按原字节补入
`reports/training/m2-smolvla-native-visual-small-2026-09-09.{md,json}`。
原报告的当时关机状态、措辞及负结果均不改写，当前核验限制以本报告为准。

## 对新计划有实际影响的发现

1. **实际训练入口是 smoke。** 旧 plan 的 `training.episodes` 已经列出 40 个，
   但 `optimizer_smoke.episodes` 与 `fixed_frame_sampler.sample_identities` 只有
   `[49, 4, 23, 43, 21, 37, 18, 34]`。checkpoint 的 resolved config 也确认只有
   这 8 个。新 40 场景计划必须修改实际生效的这两处，并保持
   `scope=bounded_visual_overfit` / phase `smoke`；独立两步 smoke 门禁仍须先通过。
   单改 `training.episodes` 或改为 formal phase 都不构成正确实现。
2. **精确继承值已核实。** seed `20260809`，batch 4，256 updates，累积 1；
   AdamW lr `1e-4`、betas `[0.9, 0.95]`、eps `1e-8`、weight decay `1e-10`、
   clip norm 10；cosine warmup 16、decay 256、末端 lr `2.5e-6`。
   保存的 scheduler 为 last_epoch 256、step_count 257。optimizer 一个组含
   500 个参数条目，其中包含冻结参数，不能称为 500 个可训练张量。
   原 Accelerator mixed precision 为 bf16，而 `policy.use_amp=false`；两字段分别保留。
   原预登记噪声为全零及 seeds `[20260905, 20260906, 20260907]`，与新计划一致。
3. **完整 reload 证据仍需补齐。** 原评估器计算完整 chunk 的聚合 MSE，但只
   保存标准空间首动作预测及汇总指标，没有保存完整 chunk 预测。
   因而原报告的“exact prediction parity”应限于已保存字段，不能推定全部
   50 步动作逐值相等。新协议要求两个空间的完整预测和独立 reload。
   原评分位于 postprocessor 前的归一化动作空间；新 evaluator 还须显式验证
   14 个有效维度、padding、相机 mask 和非视觉条件，不直接复用旧聚合值。
4. **两个历史元数据矛盾不能静默继承。** 旧 plan 同时保留了
   `formal_training_claim=true` 与 `formal_training_authorized=false`，实际执行为
   bounded smoke；新登记须如实标识，保留旧 plan 哈希。保存 policy config 的
   state feature shape 为 `[6]`，历史真实 forward 则是 state 14 / action 14，
   内部 pad 到 32。不能只凭配置 shape 判兼容，也不把这个元数据矛盾直接定为
   视觉失败原因；新真实预检须重新证明实际合同。

历史结论仍为 train 4/4、dev 0/4（cyclic 协议），五个 validation episodes
已用于开发。全非自身错配、40 场景训练、完整 reload 与任务成功均为
`not measured`。冻结 345 个 VLM 张量数值未变等参数检查来自已核验的旧报告，
本轮未加载张量重算；历史记录另列 199 个无损 dtype 转换。

## 资源与剩余门禁

当前容器查询返回 `No devices were found`（exit 6），未检测到可见 GPU；
平台开关机与计费页面未独立核验，不能仅据此认定平台模式。
软件包元数据为 Python 3.12.3、Torch 2.8.0+cu128、LeRobot 0.6.2、
Trackio 0.28.0、Transformers 5.5.4、Accelerate 1.14.0。
LeRobot 安装来源指向已登记的 `c903b114a90e703b3f7d0c46cb38727c328c55ff` archive；
没有下载、安装依赖或导入模型包。

只读清点时 durable 空闲 8,404,463,616 bytes（约 7.83 GiB）。磁盘门禁尚未通过：
需计算新 smoke、候选 checkpoint、export、原子保存临时副本和至少 2 GiB 余量。
不删除旧证据腾空间。CPU RSS 继承更紧的 10 GiB 上限；建议 GPU allocated 8 GiB、
reserved 10 GiB，仍须另获计算预算后实测。历史 131.73 秒 / 2.82 GB allocation
不代表新评估和 reload 可在预算内完成。

下一模块：在新服务器版本目录中核对本地/远端差异，补全全非自身错配 evaluator
和分阶段 hash-bound 计划，绑定实际 sample schedule、原生 loss/LR 源码与运行命令。
模型/数据缓存 payload 的完整新核验、当前真实 preflight、两步 smoke、磁盘预算、
完整预测 reload 仍待执行。现有文档保持 `executable=false`；新的计算预算未获授权。

后续唯一候选仍是 fresh base 的 40 场景、batch 4、256 steps；旧 A 只评估，
不重训、不续 optimizer，不叠加 paired loss/dropout/unfreeze。统一采用计划中的
3/4 噪声验收，分别检查关节、夹爪、训练拟合和 reload，hidden 封存。
失败保留结果并停止，不自动再训练。

本轮文档验证：九份源码与两份原报告原字节哈希、已登记哈希比较、授权边界、
seed/RSS 一致性、40/5/5 split、固定预算、相对路径与凭证模式检查通过；
新增 diff 检查通过。没有运行 ML 单测，历史 47 项回归未重跑。
