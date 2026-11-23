# 本轮三实验实现与执行边界

用户最新授权为补齐原 Exp1/2/3 的实现和真实数据，并在整轮登记任务退出、
证据回传校验后关闭计费 GPU。修订002中旧“本轮不执行”和旧成功前置条件是
历史快照；本轮独立测量不以任务成功为前置条件。安全、数值、数据真实性和
身份约束继续适用。旧报告及关机记录保留。

## 新入口

- Exp1：`scripts/diagnose_smolvla_base_five_scene.py` 与
  `configs/diagnostics/base-five-scene-independent-20260927-002.json`，独立测量
  五个原开发场景。单回合安全中止继续报告后续独立场景；不改写正式 G3/G4。
- 数据：`scripts/collect_paired_recovery_v2.py` 及 `paired-{q1,q2,bank}-20260927-001.json`，
  保留 S/R 全尝试分母、当前状态纠正、完整恢复状态和原始图像。
  `scripts/audit_paired_collection_v2.py` 与 `scripts/export_paired_cohort_v2.py`
  独立复核并生成训练接口；仅全50步连续合法监督 chunk 可用。
- 训练：`scripts/prepare_paired_training_bundle.py` 生成独立 A/B/C native-v2
  配方，`training/paired_mixture.py` 生成共同 3D+1aux 曝光表和匹配检查，
  `training/paired_mixture_runtime.py` 以显式 `paired_mixture` feature 接入原生
  DataLoader。D 元数据与 train-only normalization 保留；native loss 不改。
  每臂新初始化、2步独立 smoke、5000主更新、20000曝光、固定末步 checkpoint；
  B/C 的 aux 输入不存在时记录 input_unavailable，不能拿 D 冒充 S/R。
- 策略测量：`scripts/evaluate_paired_policy_v2.py` 绑定实际导出 manifest、完整
  tensor reload 和共同 D normalization；逐臂测五个场景，也可恢复审计 bank
  的完整分支，执行最多150步。该入口报告精确偏离状态起点的任务成功；
  几何 recovery_event 仍单独标为 not_measured，不以终局指标补造该事件。
- 生命周期：`scripts/run_round_experiment_job.py` 是新工作守护入口。
  旧 `work_only_guard.py` 的780秒上限保留，新入口按独立登记预算启动守护和
  worker、身份核验、存储/RSS/输出上限、阶段日志及真实退出记录。
  它没有平台关机入口；整轮收尾关机仍须单独检查全部登记项与回传收据。

三臂生成器接收已审计 D inventory、独立 collection cohort、原生 uniform-loss
v2 template 以及明确 camera 映射。模板含 dropout、额外 loss 或 vision-unfreeze
合同会被拒绝。非专家动作跨 chunk 不入训练。runtime 同时记录实际初始化
tensor 身份和成功完成的 native update 数；曝光表只说明计划，不能替代运行记录。

## 当前真实验证

本轮两次正式权限只读 SSH 调用分别运行环境预检和 `true`，均返回连接关闭。
未传输新源码、未启动本轮模型/采集/训练、未调用关机。连接关闭不证明平台已停机。
不能沿用较早 GPU 空闲快照作当前资源准入，也不能声称三实验已完成。

新增训练/策略测量/守护及注册源码共7文件的 WSL 只读 AST 解析通过。
原生 DataLoader/Accelerate 整合、
实际初始化对齐、真实 sampler 消费顺序、D normalization parity、CUDA smoke、
checkpoint/独立 reload、存储/耗时预算和正式评价仍须在恢复远端连接后运行。
本地 Docker 暂停，没有恢复它或在 Windows Python/WSL 主机执行 ML 测试。

## 恢复连接后的顺序

1. 新只读资源/身份预检并 content-addressed staging，运行新增测试；
2. Exp1、Q1、Q2、独立 audit/export、恢复 bank 按各自身份与工作预算执行；
3. 封存真实 N、有效 frames/slots、所有保护场景指纹和共同曝光顺序；
4. 三臂独立初始化对齐、no-optimizer benchmark、独立 smoke/reload，然后各臂
   5000主更新和固定末步 export/reload/独立测量；
5. 若缺真实输入或基础设施无法执行，保存原因并向用户报告，不冒称完成；
6. 所有本轮登记工作退出、证据回传校验、没有其他 worker 后才作整轮关机。

这里只记录本轮实现边界，不提供真实任务成功、模型修复或 M2 完成证据。
