# Hestia：真实 B/C 采集与完整性封存模块

状态：实现待 Linux 回归；尚未运行新模型或新增 optimizer update。上轮已经通过的 GPU 两步预检见 `reports/training/m2-smolvla-hestia-gpu-preflight-result-2026-09-10.{md,json}`，不能把该结果扩张为本模块验收或视觉泛化修复。

## 本模块交付

- `scripts/evaluate_visual_fit.py`：新 B/C 原生 collector，保留旧 A/B collector 原文与 256-step 身份断言。复用同一原生图像/状态/噪声入口观察方式与数值评分公式；保存七组完整数组、真实输入 trace、模型/processor/config 身份和进程信息。检查正确图与全非自身错配的**平均误差**，不先平均预测；不把目标输入 policy。
- `visual_fit_contract.py`：固定 40/5/5 split、B 256 / C 1280、native optimizer/scheduler、Action Contract 父配置及 train-only normalization、原生 feature 顺序与 frozen 范围。读实际 `train_config.json` 后生成 recipe，不能只靠元数据标签。
- `scripts/inspect_hestia_schedule.py`：原生 fixed-frame sampler 与独立 SeedSequence/randperm 的 5120 样本顺序比较。仅合成索引，无真实数据或 optimizer；每场景应恰好 128 次。
- `scripts/seal_visual_fit_candidate.py`：读取实际 C 完整恢复状态、观察器成功 update ledger 和全部 LR；逐 tensor 对比固定 base 与 C，核验冻结 VLM 和训练 expert/projector；核验四个完整 checkpoint 及 finite metric；绑定 parameter audit、完整文件清单和输入证据。
- `scripts/prepare_visual_fit.py`：从原控制计划生成 matched 1280 update / 1280 decay 的新 YAML，走真实 v2 `_resolve_plan` 回读，创建带缺失前置项的不可执行合同模板。不会继承历史 deadline 或自动授权模型执行。

## CPU/只读验收

在新 commit 独立 checkout、已登记 AutoDL Linux 环境执行，当前不启动训练，不下载模型/数据：

1. Ruff 和 `test_smolvla_visual_fit_contract.py`、`test_smolvla_visual_fit.py`、`test_smolvla_observed_launch.py`、`test_smolvla_training_observation.py`、原 `test_smolvla_visual_coverage.py` 的回归；还需原 v2 plan/launcher 相关检查。失败或 skipped 均不算通过。
2. `prepare_visual_fit.py` 在唯一新输出目录生成 YAML，并验证 native reader round trip；生成的 contract 必须仍为 draft、无 deadline/候选 checkpoint，collector 必须在模型 import 前拒绝它。
3. `inspect_hestia_schedule.py` 运行合成原生 sampler，验证完整顺序和每场景 128 次；不复用旧 1024-index 列表推定 5120 顺序。
4. 只读核对 B 的真实 `train_config.json`、processor 文件、training_step/scheduler_state 与既有报告。尤其核验原生保存格式中的 `resume`、`accelerator.gradient_accumulation.steps` 字段；当前实现预期必须经实际文件证实，缺失不能补写为 False/1。
5. 读取 Hestia 两步 smoke 的已保存原生配置，用原生配置类型解析/序列化修订后的候选预算；不把合成候选配置当作已运行 C，不读取/续跑旧 optimizer。

执行入口为 `scripts/check_hestia_collector_cpu.py`，固定上述检查次序；源文件、原生依赖和实际 B 文件清单在结果中封存。每个 CPU 检查命令最多 180 秒；本模块最多 600 秒执行预算，长进程由独立控制进程守护，退出后保留 120 秒小证据收集窗口并按用户要求关机。无卡关机必须实测 nvidia-smi 明确报告无设备，且不存在其他项目任务；这不改变原 GPU worker 的关机检查。静态源码修复和格式化不等于测试通过；首次回归前只登记为 not measured。

## 主实验仍需补齐的启动条件

本模块只补采集/封存能力，不自动启动主炉。主候选必须使用已登记 `m2-smolvla450m-visual-hestia-fit40-001`，新建 base/optimizer，40 frame-0 场景，batch 4，seed 20260809，1280 updates / 1280 cosine decay、warmup 16；该干预同时改变预算和 LR 路径。

尚待封存的远端事实：本轮源代码 CPU 验收、完整 saved-config 字段、实际候选配置 round trip、当前磁盘剩余空间/新增 checkpoint 预算、当前 GPU 与环境身份、与已通过两步 smoke 的原生源码一致性、主 supervisor 的 PID/启动时刻/期限。不得从旧 SSH 端口、旧关机记录或草稿字段猜测这些事实。

完整主 worker 还须串联训练前 doctor/benchmark/资源检查、实际观察器、checkpoint 封存、B/C 各两次独立采集、七数组 reload、历史 B 原始数组复现与最终比较。只有这些条件完成并有明确执行授权才可运行；不存在因 collector 已能评分而自动开炉的路径。

验收保持 C 的 train40 与 dev5 关节/夹爪/aggregate 在**相同至少 3/4 条件**下正确图优于平均错配且低于均值下限；开发改善与标准 full-chunk/first-action 非回退；七数组独立 reload 完全一致。B 的已知夹爪负结果保留，原 aggregate 训练拟合前提保留。五个开发场景不是独立测试，hidden 仍封存。失败保留负结果并停止，不自动叠加 paired loss/dropout/unfreeze；M2 仍需闭环验收。
