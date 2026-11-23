# 原D缺失完整场景状态后的研究边界选择

原D元数据能力检查已经由独立执行代理在现存固定revision缓存实际完成：`paired-source-provenance-20260927-001/source-provenance-capability.json`。它核验原receipt、meta/info与episode/data Parquet schema/footer，没有读取任何原始轨迹row或image；simulator seed、scene/parent、object pose、完整integration四类候选字段均为空。源artifact仍在远端，回传与完整文件SHA核验由执行代理完成；本文件不假称已在本地独立核验原报告字节。

这说明已检查来源没有可直接读取的完整reset/物体状态或生成seed。14D motor observation无法唯一推出物体自由关节、qvel、warmstart、ctrl/act、mocap、RNG和wrapper状态。单一有损AV1 top视频不提供这些隐藏状态；未记录的完整状态不能由一个新hash重构。不能据此宣称整个源系统永远没有其他历史原始HDF5/采集日志，但当前已核验缓存没有严格库存所需字段。

## 选择一：保留原严格准入

继续新的003-B真实试采、完整状态/事件校验和普通/恢复尝试分母。只有找到原始采集reset/场景日志，或经另行登记的可核验重建取得原D/dev/hidden完整场景库存，才可将成功候选用于原严格Exp2/3训练比较。对称sampling修订不会绕过这条要求。科学成本是本轮可能只能完成真实数据工程和负结果，不能把strict N=0变成三臂比较完成。

## 选择二：明确授权有限来源谱系探索

可以开展一个可执行且诚实的新比较，但必须登记为**新的有限范围研究**，不能冒称原严格场景隔离已通过。需要用户明确同意：

> 允许在不读取、使用或按原dev/hidden观测调参的前提下，用独立新生成的同域S/R parent家族、完整新reset状态和新场景/恢复bank隔离，开展固定开发协议的探索性Exp2/3比较；承认无法证明这些新物体初始状态与原D/dev/hidden全部无重合，不把结果解释为原隐藏集无泄漏验收或部署泛化。

落实边界：

- 原D仍仅从已核验train40导出读取，normalization仍D-train-only，原dev/hidden不提供新teacher动作、恢复标签、采样调参或模型输入。没有借机读取hidden视频/初始观测。
- 新S/R完全由冻结teacher代码和新注册seed/current-state动作生成，代码路径不读取原D观测/action/video。每个新parent保存完整reset integration、wrapper/RNG、几何指纹、代码/模型/合同身份；S/R后代同一split。
- 新Q1、正式候选、teacher调参、原policy开发Gate和独立恢复bank使用不同已登记来源家族。新源之间检查实际完整状态/几何重复，保留未测与失败分母，拒绝未知label、故意错误动作chunk及不满足恢复→终局success的候选。
- 003-B对称合法起点网格仍按已批准规则冻结；原始帧、原始有效slots差异、所有排除及重复曝光完整披露。训练、原动作/native loss、D-only统计和固定最终checkpoint不改。
- 字段必须区分 `N_strict=0`、`N_exploratory`、`fresh_lineage_verified=true`、`legacy_object_scene_nonoverlap=not_established`。不能继续写 `scene_disjoint_verified=true` 表示与原D完整场景已排除；原strict模式和验证器保留。
- 报告只回答：该已生成、真实成功且对称匹配子集在固定计算量下，对五个固定开发场景及独立新恢复bank的影响。它不估计训练方差、不代表全恢复尝试、不代表原hidden集或部署泛化。没有真实成功匹配子集则探索N也为0，不能制造BC数据。

该选择实质改变原要求的“跨原D/dev/hidden完整scene/state隔离”验收范围，不能作为普通代码修复默默执行。用户已选择003-B仅授权sampling与grasp触发修订，尚不自动授权本研究边界变更。应取得上面明确选择后，使用新计划/run身份，将探索模式标记与数据/训练/评价配置一起封存。Q1与独立base測量可以继续；任何单项失败或选择等待都不触发关机。
