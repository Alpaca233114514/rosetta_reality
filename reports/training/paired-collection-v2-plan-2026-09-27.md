# 配对采集 v2：有限独立试采、正式候选及恢复 bank

用户最新授权覆盖修订002的旧“本轮不采集”快照，并取消独立科学测量之间的任务成功前置依赖。本计划允许当前实验性 teacher 的真实独立试采，不声明其完整资格通过；动作、身份、数值、物理安全和数据真实性标准保持。任务失败只结束该回合，身份/基础设施/非finite/资源异常停止该作业。任何阶段都不触发平台关机。

入口为 `scripts/collect_paired_recovery_v2.py --plan <registration> --output <new-directory>`，独立文件复核为 `scripts/audit_paired_collection_v2.py --root <collection> --output <new-audit-json>`，训练清单转换为 `scripts/export_paired_cohort_v2.py --root <collection> --audit <audit> --camera-map <explicit-map-json> --output <new-cohort-json>`。集成代理独占远端重活调度，外部 work-only guard 必须覆盖作业。

| 阶段 | 登记 | 固定场景 | 主尝试/总主步上限 | 工程恢复验证步 | 工作/证据上限 |
|---|---|---|---|---|---|
| Q1 | `configs/diagnostics/paired-q1-20260927-001.json` | 2100–2103 | 8/4000 | 最多16 | 1800秒/256MiB |
| Q2 | `configs/diagnostics/paired-q2-20260927-001.json` | 2200–2219 | 40/20000 | 最多80 | 7200秒/1GiB |
| bank | `configs/diagnostics/paired-bank-20260927-001.json` | 2300–2309 | 20/10000 | 最多60 | 3600秒/512MiB |

来源固定 `mink-coupled-translation-teacher-20260927-001`。各四场景按 approach正/负、align正/负顺序登记；扰动仅为原14D合同 `left_waist` 的 ±0.04rad绝对命令，持续1个50Hz环境步；不改物体、奖励或物理参数。实际扰动后该维状态变化至少0.002rad才记为已测偏离。每条主轨迹总计最多500步，R纠正最多150步并计入500；没有触发、拒绝、超时、安全中止都保留，没有补抽或自动重试。每次持有动作必须原合同合法且无需clip；工程恢复检查出现物理异常不继续执行第二次。

R恢复条件事先固定：approach分支双物体均有抓取contact；align分支双抓取并且有peg/socket contact；连续5步满足。还必须有真实扰动、从下一步开始根据当前几何纠正、恢复早于原任务 `info.is_success` 真值事件。reward上升不是成功标签。失败/未知或故意偏离动作都无正监督。仅完整连续50动作全合格chunk可提供BC输入；末尾不补padding，不修改native loss。S/R以有效输入帧及完整监督slots残差≤5%匹配，没有重复帧、插值、删难段或增加失败seed。

每对从完整 `mjSTATE_INTEGRATION` 与Python wrapper/RNG快照恢复同一reset，而不是仅拷贝qpos。reset和R事件前各执行同一持有动作两次，字节级比较integration、wrapper、几何、native图像哈希和原奖励/安全字段；bank还独立验证实际扰动后的完整分支恢复。持有验证步单独计入工程分母，不进入标签。原始无损uint8 HWC相机以逐帧NPZ保留，trace/label关联50Hz PTS、标准action、真实执行后state/contact/outcome。`restore_paired_branch_v2.py`可在同源同模型环境恢复bank的完整integration与Python/RNG receipt，并做恢复后receipt严格比较；真实跨进程验证仍需远端运行。

scene指纹绑定reset机器人、物体和EEF几何，另外保存完整integration hash。parent/pair是统计和split单位。Q1永不进入训练和bank，bank独立于Q1/Q2、teacher调参、原policy Gate。seed清单只排除已知重复编号，不能证明与原D/dev/hidden物理状态完全独立。当前三个登记明确 `protected_scene_inventory_complete=false`；必须将真实保护状态指纹审计绑定到新身份，才可准入训练。因此输出区分 `candidate_matched_N` 与最终 `N`；N=0保留完整负结果，不制造训练输入。保护库存不完整不阻断这批独立物理测量。

独立复核重算全尝试分母、连续行/PTS、真实success与恢复事件链、全部图片/文件hash、正监督mask、跨错误动作chunk拒绝、有效slots和匹配。数据完整性复核不把实验性来源改成全gatepassed，有限成功匹配子集效果也不代表全体尝试。closeout为manifest和独立审计，平台生命周期仍由整体Exp1/2/3负责人管理。

实现静态AST检查已通过；本计划写入时尚未执行真实配对采集、loaded-state恢复、bank跨进程恢复或训练。新增纯文件测试必须在授权Linux容器/AutoDL执行，不用Windows Python或本地ML环境。
