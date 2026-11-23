# ALOHA Q阶段远端执行观察（2026-09-27）

身份：`aloha-qualification-20260927-002`。本轮用户重新提供已开SSH，授权执行此前本地准备并要求一次性上传、看门狗、合理截止与关机。实际执行范围为两种已下载ALOHA来源的Q资格库存审计，无模型、optimizer、仿真或下载。

## 部署身份与生命周期

经 `scripts/stage_autodl_from_wsl.sh --verified-source-delta` 创建新版本workspace，旧base、任务和失败证据保持原样：

- base composite SHA：`8684002da746383fe085a7ddfb48c15d8c73d1874296b935884d0e443d0bee0b`。
- source delta archive SHA：`6c88392fb1c15326973f6a6d564695950b7531f9bff3e1d5863650ac7f4e78e5`。
- 新workspace：`workspaces/20260927T080322Z-66a5bbcdbdcd-6592b19037ad`。
- 新composite SHA：`6592b19037ad8974c6c4d3548751496aeba4cbdc4e8fee5cfce9077a6a751967`。
- 14个上传源文件的远端SHA与本地一致后才启动，工作截止北京时间16:07:26，独立guard关机截止16:09:26，平台兜底16:10。正常完成预留最多180秒回传窗口；没有自动后续训练、retry或resume。

远端预检实测RTX4090D、24564MiB；cgroup内存64,424,509,440 bytes，数据盘约11.96GB可用，没有其他任务。默认Python缺PyArrow，改用现成`smolvla-cuda-001`环境，实测Python3.12.3/PyArrow25.0.1/NumPy2.2.6，没有依赖安装。新的worker按阶段检查RSS及证据预算，设置每文件8MiB上限、1GiB RSS检查和1GiB磁盘余量门槛；主工作有独立guard绝对截止。profile仅作为runtime参考，不冒充通过了模型CUDA doctor。

预检另确认human原下载receipt没有顶层revision，版本在`source_files`。本轮修复新审计入口的兼容解析，要求全部source_files绑定同一个明确dataset/revision；缺失或混合版本仍拒绝。旧receipt不修改。新增三项回归在当前本地Docker人为暂停时未执行；远端登记任务的完整标准库回归phase报告exit0，原始日志仍待完整取回核验。

## 已观测结果与证明范围

远端新`result.json`及`handoff-ready.json`已通过SSH读取一次，远端报告：

| 项目 | 远端报告值 |
| --- | --- |
| regression / inventory phase | exit0 / exit0 |
| source files / bytes整文件重读校验 | 21 / 162,343,250，均匹配source receipt |
| data parquet / row groups | 7 / 7 |
| raw row groups读取 / 排除 | 0 / 7 |
| raw frames / hidden loaded | 0 / false |
| optimizer updates / rollouts / model loaded | 0 / 0 / false |
| worker elapsed | 0.958925秒 |
| qualified S/R cohort / training_ready | false / false |
| sealed handoff members / bytes | 14 / 35,773 |
| evidence manifest SHA | `570c094a1badbc295abab227312cfd04d1590fa25eb38ab110f26b8a2464ab97` |

这次确实执行了真实source整文件hash及Parquet footer审计，但**没有读取raw训练样本**。不能用source hash通过或metadata同shape/fps证明成功/恢复标签合格。

两来源都声明ALOHA、14维motor ordering与50Hz；human有50episode/25,000frames，scripted有50episode/20,000frames。实际row groups为human `[0,14]`、`[15,29]`、`[30,44]`、`[45,49]`，scripted `[0,17]`、`[18,35]`、`[36,49]`；每一组都与原dev/hidden编号相交，因此依现有合同整组跳过。raw行/视频对齐、成功结局、偏离→状态条件专家correction→恢复链、protected scene/parent身份与匹配N仍未确认。

data schema仅见state/action/episode/frame/timestamp/next.done/index/task_index，没有专用干预或成功恢复链字段。这只能说明源文件没有提供这些显式列，不证明真实轨迹从未恢复；`next.done`不能当成功标签。原human仍属于D，不能计新增N。

## 回传失败与平台状态

原始审计结果已落durable namespace `runs/aloha-qualification-20260927-002`。回传连接被关闭，本地`reports/training/aloha-qualification-20260927-002/results.tar`为0字节并保留，14个artifact成员的本地独立SHA核验**尚未完成**。本地`remote-result-observation.json`只是上述stdout观察转录，不能当已验证的原始artifact副本。

同一实例的新平台页面已显示“已关机”，证明此轮没有继续保持GPU运行；SSH断开不是关机证明。guard shutdown-request及原始regression/inventory日志尚未取回，不能将平台已关机单独归因于guard程序。本次没有手动平台关机请求，平台另有定时关机兜底。

为取回已完成证据而尝试无卡模式启动时，自动审核拒绝，理由是它与先前用户要求有卡开机不一致且必要性/授权未明确。没有绕过审核或重启GPU。现已向用户提出限定5分钟无卡恢复证据的明确批准请求；在批准前保持实例关机，不重跑审计。

## 后续技术缺口

先完成原始结果回传与独立SHA，再准备能在读取state/action/image前排除hidden的隔离reader；不得对mixed row group直接放宽门槛。视频、标签及protected scene/parent清单需真实审查才能确定S/R与N。完整row qualification、matched cohort和Exp2/3训练仍未完成。Exp1现成有效base Gate3前置证据仍缺，不在本轮重跑或修改Gate。M2状态不变。
