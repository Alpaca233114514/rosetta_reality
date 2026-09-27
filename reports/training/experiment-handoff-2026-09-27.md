# 实验交接（2026-09-27）

交接身份：`experiment-handoff-20260927-001`。用户要求：**做完实验交接，先不启动任务**。

| 门禁状态 | 值 |
| --- | --- |
| execution_authorized | false |
| task_started | false |
| training_authorized | false |
| evaluation_authorized | false |
| launchable | false |
| automatic_wakeup_or_schedule | none |

这些状态指本交接涉及的实验任务；主任务已获授权的数据下载独立进行，不因此停止。此交接不授权开新聊天/任务、调度、SSH、模型或数据加载、ML测试、训练、评估、仿真、采集、适配、提交或推送。下一轮先静态核对交接与现成依赖，**禁止自动执行**。

## 入口与已完成工作

- [三项实验设计](three-experiment-design-2026-09-27.md)；[不可执行设计 JSON](../../configs/diagnostics/three-experiment-design-20260927-001.json)。两者已完成静态检查，没有实验结果。
- [数据下载报告](targeted-dataset-staging-2026-09-27.md)：来源版本、子集、下载与最终验收以该报告及收据为准。
- Exp1 [原计划](m2-smolvla-prepost-gate4-plan-2026-09-24.md)、[结果取回记录](m2-smolvla-prepost-gate4-retrieval-2026-09-24.md)、[最新综合分析](m2-smolvla-prepost-analysis-and-next-plan-2026-09-24.md)、[独立核验 JSON](m2-smolvla-prepost-analysis-verification-2026-09-24.json)。
- 稳定入口：[AGENTS.md](../../AGENTS.md)、[架构文档](../../docs/m2-smolvla-architecture.md)。涉及后续 SmolVLA 工作，仍需按其中阅读要求核对 Faust/Zen 审计与相应最新完成报告。

当前功能分支为 `codex/prepost-parameter-comparison-20260923`，已有大量未提交修改，应全部保留。设计只新建两份文件，本交接只新建本文件；没有改变架构、模型、Gate或历史证据。M2仍未完成。

## Exp1：唯一待办与依赖边界

用户明确修正：**“实验一只剩未训练模型的 gate4 没跑，其他原样。”** 唯一待办为 pinned pretrained base（未做本项目任务训练）的 Gate4。原配置、流程、共同 ALOHA 接口、processor、normalization、噪声、Action Contract、五seed/500步、阈值与训练后结果全部原样；不新增训练或重跑训练后部分。

须先核对**现成有效 base Gate3**报告与 artifact/code/simulation-plan 同源绑定。已读9月24日历史报告记录 base Gate3失败、base Gate4未测；trained Gate3通过、Gate4为0/5。这与用户的当前待办说明分开记录：不改写历史，不假定新Gate3已通过，也不据旧失败自行新增修复、Gate3重跑、离线采集或诊断轴。有效依赖未找到则保留 `pending_verification`，不得绕过门禁或把base Gate4填为0/5。

Exp1现成 Gate3证据路径/hash及同源绑定：**待核对**。当前仍不启动base Gate4。原历史training-effect证据有共同接口和dtype/初始化解释边界；完整成功率之差尚不可计算。

## Exp2/3：保留设计，尚不可训练

Exp2比较A（原数据D）与B（D+targeted recovery）；Exp3按用户选择比较C（D+N条普通成功轨迹）与B（D+N条recovery/correction轨迹）。B仅在全部数据/训练/评估身份相同时共享，不算两次独立重复。

草稿建议每臂batch4、5000个**成功 optimizer updates**、同seed和固定最终checkpoint；每batch共用3个D slots，辅助1个slot分别来自D/R/S。故A有20000次D exposure，B/C各15000次D+5000次辅助exposure。Exp2估计固定算力下的混合替代效果，包含原数据曝光下降；Exp3严格共享base顺序、曝光与权重，并匹配N、有效输入帧、有效监督slots、长度与padding，较好区分新增数据内容。不能仅按epochs、文件数或重采样次数说预算相同。

N尚未决定（建议上限20）；5000 updates/臂、三臂至多15000主updates及独立smoke上限6步均是**未授权建议**，不是已登记run。实际cohort、初始化/采样/代码/processor/runtime hashes、恢复bank/阈值、wall/存储/证据预算与资源benchmark均未封存。历史A只有全部身份匹配时才可复用；否则只是参考，需要的新A属于Exp2/3后续登记，不扩展Exp1。

先证明同域数据有真实 deviation→状态条件专家correction→恢复/成功链，以及普通成功对照、scene/parent-trajectory隔离与无泄漏。新下载同revision ALOHA human不是新增轨迹；失败动作、缺标签或偏离后的时间索引专家动作不能冒充正确恢复监督。现有teacher尚无完整gate通过依据；不能伪造oracle gate状态来接入人工纠正。

当前没有已验证的同域S/R matched cohort。REBOOT为WidowX follower 14维/30 Hz，当前ALOHA为14维绝对关节/50 Hz；维数相同不代表兼容。Sirius/Can Paired/MimicGen也保留独立合同。跨机器人数据仅是后续资格研究候选，不拼入ALOHA、不用于凑N；如须新采集/teacher/适配，另行授权并登记。下载完成不等于可以训练。

## 数据交接：进度快照与最终状态区

相对namespace：`datasets_external/targeted-20260927-001`（相对远端Rosetta数据盘根）。旧批次为 `datasets_external/targeted-20260925-001`。不经桌面中转数据。

以下为主代理提供的**交接准备时进度快照**，并非本交接独立实测或最终验收：累计约52.03 GB；ALOHA human 11文件/91,348,537 bytes、MimicGen 5文件/4,851,403,523 bytes、Sirius 67对象/7,031,694,115 bytes、USB-A 82文件/37,283,634,402 bytes已传输；RJ45 47文件/16,900,694,536 bytes仍在下载。全新增选择为212文件/66,158,775,113 bytes。RJ45选24episodes/21,559 frames仅经元数据审核及视频引用检查，真实行和视频解码未验证。原全量stats保留，未来显式subset/train split应重算训练统计。

旧scripted+Can Paired共12文件/171,563,018 bytes已重验。RACER、AgiBot及其余RJ45数据分片因容量延后。当前实例无卡，曾实测cgroup内存2 GiB，不证明下一轮GPU资源可用；剩余约12 GB只是规划估计。

**下表为主代理完成下载及关机后的最终实测。** 收据名称均相对上述新namespace。本地副本在 `reports/training/targeted-dataset-staging-20260927-001/`；平台关机证据另在该本地目录保存。

| 最终字段 | 当前待填状态/证据入口 |
| --- | --- |
| final_transfer_status / files / bytes | complete；212文件/66,158,775,113 bytes；0失败、0剩余partial；数据留在远端 |
| final_independent_whole_file_verification | passed；全部212文件的源校验和独立整文件重读通过，见 `independent-verification-stream.json` |
| worker_result_and_exit | `result-stream.json` complete=true；`exit-code-stream.txt` 为0；worker_running=false |
| provenance_receipts | 已封存上述收据及 `closeout-observation.json`；旧12文件/171,563,018 bytes也已重新核验 |
| receipt_hashes_and_retrieved_copy | 14个元数据文件/404,710 bytes已取回；22项收据/代码SHA比较通过；`provenance.sha256` SHA=`8f4fefad6f3955decba6c725a0122c2350676c697dc9e655371c7bb65ab3da2d`；各主收据hash见[下载报告](targeted-dataset-staging-2026-09-27.md) |
| final_disk_bytes_total_used_available | 91,268,055,040 / 79,312,007,168 / 11,956,047,872 bytes；观测时间 `2026-09-27T03:41:21.273226+00:00` |
| final_memory_and_worker_state | worker已退出；cgroup限额2 GiB，anon321,306,624 bytes，OOM/OOM-kill均0；仍非GPU准入证据 |
| platform_shutdown_requested | 已通过平台页面确认；请求窗口 `2026-09-27T03:43:30Z` 至 `03:43:52Z`；未请求释放实例 |
| platform_shutdown_independently_verified | 已确认；重新读取平台页面，`2026-09-27T03:44:55Z` 同一授权实例显示“已关机”；见本地 `platform-shutdown-verification.json` 与本聊天关机截图 |

此区填写完成也只关闭下载交接，不改变顶部实验 `execution_authorized=false` / `task_started=false`。

## 下一轮的无执行接续顺序

1. 先读本交接最终状态区、下载报告和两份设计，确认哪些是完成证据、哪些仅建议；保留原工作树与失败材料，不从旧报告的pending列表自动续跑。
2. 仅静态核对Exp1现成Gate3报告和原身份链；缺失则记录确切缺口，保持base Gate4唯一待办，不自行修复或启动任何实验。
3. 核对下载最终收据与来源身份；将真实数据行/视频/标签审计、S/R匹配、恢复bank及采集适配列为尚未执行的资格工作。数据审计加载和ML检查需另行明确范围与运行环境，不在Windows Python或WSL主机Python执行。
4. 用户另行授权具体阶段后，才准备新的create-only执行登记，补齐真实hash、完整采样/更新计数、资源/磁盘/截止及停止条件；登记前后均不得降低既有Gate、安全基线或Auto Review。

没有一键启动命令、自动唤醒或后台实验。离线改善、训练完成、文件hash通过、20步安全通过或Basin导入，均不能替代完整任务成功及原Gate验收。
