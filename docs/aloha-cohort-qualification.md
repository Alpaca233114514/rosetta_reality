# ALOHA 成功／恢复数据资格工具

此入口为 `three-experiment-design-20260927-001` 的 Q 阶段准备：先查真实数据与证据，再封存 S/R 配对。它不训练模型、不修改原 Action Contract、不转换机器人、不接入现有 oracle recovery manifest。

## 两个独立步骤

### 1. 源文件及行结构核验

入口：`scripts/audit_targeted_aloha.py inventory`。本地登记为 `configs/diagnostics/aloha_qualification_20260927_001.json`，使用已取回的最终下载manifest SHA固定 ALOHA human 来源。该human来源是原数据D，不能充当新增S/R或重复计N。scripted来源的完整receipt未在本次登记中封存，暂不加入可执行来源列表。

文件/数据只在已登记Linux容器中读取。以下示例中的 `DATA_ROOT`、`OUTPUT_NEW` 由该容器调用者提供，输出必须不存在：

```bash
PYTHONPATH=src:. python scripts/audit_targeted_aloha.py \
  --output "$OUTPUT_NEW" inventory \
  --plan configs/diagnostics/aloha_qualification_20260927_001.json \
  --data-root "$DATA_ROOT" --read-train-rows
```

不传 `--read-train-rows` 时，只读取Parquet footer/schema/episode统计元数据。传入后，只有 episode statistics 完整且整个row group属于train40的组才可流式读取，batch128；任何共享dev/hidden、缺min/max或含null的组整体跳过。跳过组与未知组保留在报告，不能说已经审核全部轨迹；若源将大量episode混入一个row group，需先另外设计保证隔离的reader，不能放宽hidden读取条件。

已读取部分检查state/action有限性及配置维数、frame和50Hz时间连续性。维数相同不是物理合同通过，行数观测也不是完整episode和有效监督slots证明。视频对齐、合法范围/单位/ordering、成功/恢复标签及场景身份均保留为未核验。此工具不推断“done=成功”或“没有介入列=普通成功”，不使用源全量stats训练，也不下载任何文件。

### 2. Hash绑定资格报告及配对验证

入口：`scripts/audit_targeted_aloha.py cohort`；模块：`src/rosetta_reality/diagnostics/cohort_qualification.py`。

```bash
PYTHONPATH=src:. python scripts/audit_targeted_aloha.py \
  --output "$OUTPUT_NEW" cohort \
  --manifest "$COHORT_INPUT" --evidence-root "$EVIDENCE_ROOT"
```

`COHORT_INPUT`不是训练manifest，必须包含：

- `action_contract_sha256`：当前固定合同；`protected_groups`：D、dev、hidden、Gate、teacher/tuning及其scene/parent/collection_family/content身份，清单需完整。
- `candidates`：逐trajectory的dataset、40位revision、episode、role(S/R)、train split、scene、parent_trajectory、collection_family、64位content_sha256及`already_in_D=false`的可审计依据；effective_frames、valid_target_slots、padding_fraction、duration_seconds及task/embodiment/scene_stratum/initial_pose_stratum/collector。
- 每条候选的`contract_evidence`、`rows_evidence`、`labels_evidence`，均为相对path与真实SHA256；报告必须逐dataset/revision/episode绑定。
- `pairs`：预先指定的一对一索引 `{ "S": 0, "R": 1 }`；至多20对、每条trajectory只用一次，不根据后续模型结果选取。

合同报告必须有已核验的物理语义等价、Gate1/2与对应合同hash。rows报告必须有完整有效帧／监督slots／padding数值、源records SHA、有限state/action、hidden未读取与视频对齐证据。labels报告必须有明确可复核的最终success；S须明确核验普通成功且不存在恢复段。

R额外要求偏离、专家纠正起止、恢复、成功帧组成有序链，连续上下文和状态条件专家依据，`time_indexed_replay=false`。人工纠正与qualified teacher明确分列；teacher需要另一个hash绑定的通过gate报告，不能借用人工身份绕过oracle门禁。当前实现只核验报告声明与文件身份，不会独立证明报告内容真实或teacher gate已与全部恢复事件逐状态绑定；这些科学证据仍需审查。

保护清单中的scene/parent/content/collection_family或同源episode碰撞将被拒绝；候选内部重复parent/content拒绝，train内部共享scene保留相关性，不能把共享scene的轨迹当独立统计重复。匹配要求预先固定的任务、机器人、场景/初始位姿stratum与collector相同，帧/有效slots/时长相对差不超过5%，padding绝对差不超过0.05。缺失、hash漂移、非法相对路径、失配或标签unknown均不准入。

即使所有候选与配对通过，输出仍为 `training_ready=false`：后续必须封存sampler、真实初始化/processor/runtime、资源benchmark与独立恢复bank。现有 `RecoveryDatasetManifest` v1不改变，也没有填写伪造 `oracle_gate_status=passed` 的适配路径。

## 修复后的独立关机guard

入口：`scripts/autodl_bounded_guard.py --job "$JOB_NEW"`。仅在用户已授权远端任务、平台独立定时关机已核验且新create-only job已登记后部署；本地验证不调用真实shutdown。

登记要求：`shutdown_authorized=true`、`started_unix`、`work_deadline_unix`、`shutdown_deadline_unix`，工作截止必须早于/等于关机截止，最长24小时；实际时长须由对应实验阶段登记，24小时是配置拒绝上限，不能当默认预算。worker必须以独立process group启动，并在`worker-pid.json`登记pid和Linux `/proc/PID/stat` start_ticks。

guard启动后用monotonic计时防止时钟回拨延长费用窗口；退出进程作为正常状态，PID/start_ticks或process group不匹配不发送信号。TERM之后再核对身份决定KILL；cleanup异常写新失败报告，仍进入关机安全检查。关机仍要求固定平台wrapper hash、Trash不存在、无GPU worker、无无关tmux/project worker；失败时保留`shutdown-blocked.json`，由平台独立定时关机兜底。

必须保持平台定时关机这一独立层。guard正常退出、SSH断开或shutdown-request都不证明平台已停机。旧 `.work/execution-readiness-20260927-001/guard.py`、原归档与失败日志保持原样，本脚本是新修复版本；本轮没有上传或实测远端关机。

## 本地验证

`scripts/run_qualification_local_tests.sh`要求传入固定本地镜像digest和新`.cache`输出目录。在WSL Bash发起Linux Docker，网络关闭、只读源码、2CPU、512MiB、无额外swap。现有登记镜像在当前引擎不可见，测试使用本机dpkg校验过的Debian Python3.13标准库离线打包镜像，不安装项目依赖。

回归覆盖进程退出/PID复用/进程组隔离、退出检查失败仍进入关机安全检查、关机授权与wrapper漂移、证据不可覆盖，以及unknown/失败恢复/replay/teacher gate/合同/泄漏/hash/path与配对条件。纯合成验证不替代真实Parquet/PyArrow、视频、标签、模型、Gate或AutoDL生命周期验收。
