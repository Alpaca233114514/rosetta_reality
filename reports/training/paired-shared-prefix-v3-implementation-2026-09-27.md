# 003-B 真实共同前缀分叉实现

旧 Q1 `paired-q1-003b-explore-20260927-001` 的 8 次尝试、2088 次真实仿真步与 N=0 原结果保留。2411 的普通成功和恢复成功没有被追认成配对训练数据；两个独立 reset 续行的完整物理/动作前缀相同，但真实原相机像素在部分帧有末位差异。具体证据见 `paired-q1-prefix-difference-20260927-001.json`。

新协议 `shared_real_prefix_causal_fork_v3` 是对已批准 003-B 精确匹配要求的实现修复。主代理已审阅同意：一次真实共同历史只采集一次，在预登记首次 grasp/align 决策点保存完整环境 integration、Python/wrapper/RNG、原 uint8 图像及状态；普通 S 与扰动 R 从同一保存的分支点继续。共同历史的图片由两条逻辑轨迹引用同一原始采样文件，不能把旧独立 R 的图片替换成 S 图片。分支后的原始 render 分别保留，不能声称它们始终 bitwise 相同。

教师原生 C solver 不做 deepcopy 或伪序列化。每条分支使用新教师，输入已保存的同一 geometry 决策历史，逐次核对 action 与 phase 精确一致来重建教师状态。该重建的 decision calls 单独记录，不算环境步。分支首行显式消费保存的真实观测：其状态与原 uint8 内容必须与捕获身份一致。

工程 `paired-shared-prefix-fork-engineering-20260927-001` 已真实完成并经独立验核：175 步共同前缀、两个 16 步无扰动续行，共 207 次真实仿真步；352 次教师重建决策逐一一致，16 对续行的 action/phase/before/after/integration/reward/done/info/safe 九类字段全部精确一致。本次 16 对续行原图像也精确一致，但这一观察不外推为渲染确定性保证。工程不包含故意扰动，不是训练 cohort，N=0。详情见 `paired-shared-fork-independent-verification-20260927-001.json`。

新入口：

- `scripts/collect_paired_recovery_v3.py`：保留原 v2 入口；一次真实 prefix，再恢复并重建两条分支。记录完整尝试分母、真实错误动作、150 步纠正上限、500 步逻辑回合上限、完整成功/恢复事件，原始标签与全部失败证据。
- `scripts/audit_paired_collection_v3.py`：继承原保存文件、恢复窗口、监督与合法 chunk 审计，并独立验证共享原历史、teacher-history 身份、分支首行原 NPZ 图像/state、完整 fork snapshot 和步数。
- `scripts/verify_paired_numerics_v2.py`：全行复算命令日志/trace/labels、每维扰动、实际关节变化、clip、纠正索引及五步接触窗口；不靠摘要阈值自证。
- `scripts/export_paired_cohort_v3.py`：必须有绑定实际文件的 v3 审计；保持训练 runtime 行/图像/native mask 和 D-only normalization 接口。

物理步数由一次共同 prefix 加各自分支实际续行及工程恢复 hold 构成。逻辑 S/R 轨迹各含共同历史，因此完整通过时独立核算 `physical_steps = logical_trajectory_frames - shared_prefix_physical_steps + engineering_restore_steps`，不把共同 prefix 双计为真实执行。环境步骤与教师决策步骤分别计数。

新 Q1/Q2/bank 计划使用 `paired-*-shared-prefix-v3-bound-20260927-001.json`，绑定已独立实测的完整新场景 registry/audit 和 lossless gzip 成本证据。原 scope 不变：`N_strict=0`，只有满足完整真实成功、恢复、min50 共同合法 chunk 及新谱系隔离者可计 `N_exploratory`；`legacy_object_scene_nonoverlap=not_established`。原 D 仅 train40，不读取或调参原 dev/hidden。teacher 不冒称 qualified，D/native loss 不改变。

此报告记录实现及已完成工程证据；新 v3 Q1/Q2/bank 的实际结果须由各自新身份输出与独立验收确认，不能以脚本或单测通过冒充数据完成。
