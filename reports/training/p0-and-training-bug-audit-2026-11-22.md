# P0 修复与训练程序 Bug 审计报告（2026-11-22）

- **基准分支**：`codex/paired-003-evidence-audit-20260930`（未提交、未推送，保留所有已有用户改动与历史证据）
- **关联规划**：`reports/training/next-step-experiment-report-and-direction-2026-11-22.md`
- **边界声明**：本报告仅记录静态审计、create-only P0 诊断工具修复、训练代码定点修复以及宿主机 stdlib 回归测试结果；本地 WSL 未启用 Docker 集成，AutoDL 远端实例保持关机，所有涉及 `torch` / `lerobot` / `mujoco` / CUDA 的端到端运行均**尚未执行**，不得将本报告中的 stdlib 单测通过等同于 CUDA 验证或 Gate 3/4 结论。

---

## 一、P0 修复清单（Create-Only，不破坏 2026-10-04 历史哈希绑定证据）

### 1. P0-A：修复 2445 单步边界探针（`scripts/paired_boundary_pilot_v2.py`）

原脚本 `scripts/paired_boundary_pilot.py`（SHA-256 `42d2b4f3aca7b4618f9f795317c90304cbd80c1710e3e65097f9a4c6e9396b12`，已原样保留，未改动）与计划文件 `configs/diagnostics/paired-boundary-pilot-20261004-002.json` 存在 5 个会在真实 CUDA 首次落盘时必然触发崩溃或漏检的缺陷。新建 `scripts/paired_boundary_pilot_v2.py`（SHA-256 `9b2a5dfffbc65ac4e653345ec5b4a5be78608cd47e3e46be9e64d7d5ac5a8fc6`）与单测 `tests/test_paired_boundary_pilot_v2_stdlib.py` 逐项修复：

| 编号 | 原位置 | 根因与实际后果 | v2 修复方式 |
|---|---|---|---|
| **P0-A1** | `scripts/paired_boundary_pilot.py:173-175, 334, 380` | `capture_physics()` 直接返回 `numpy.ndarray`（`qpos/qvel/ctrl/qacc/qfrc_*`），落盘调用 `rosetta_reality.eval.gate_diagnostic_io.write_tree()`（`src/rosetta_reality/eval/gate_diagnostic_io.py:74-100`），该函数仅接受 `torch.Tensor` / 基础容器 / 标量，传入 `ndarray` 必然抛出 `TypeError: Unsupported evidence value: ndarray`。 | `capture_physics()` 改用 `torch.from_numpy(np.ascontiguousarray(...).copy())` 落盘，彻底消除 `ndarray` 序列化崩溃。 |
| **P0-A2** | `scripts/paired_boundary_pilot.py:375, 380` | `env.step()` 返回字典含 `raw` 字段（内含 MuJoCo `agent_pos`、`pixels["top"]` 等 `numpy.ndarray`），原脚本把 `returned` 整棵树传给 `write_tree()`，同样在 step 1 落盘时必崩。 | 新增 `sanitize_observation()`，保留可序列化张量/数值字段并剔除 `raw` 原始 numpy 缓冲区，仅对 `raw` 记录摘要元数据。 |
| **P0-A3** | `scripts/paired_boundary_pilot.py:186-217` | `plain_config()` 遇 `Path`、`torch.dtype`、`torch.device`、`set`/`frozenset` 会抛 `TypeError`；`processors()` 只读取 `step.stats`，漏掉 LeRobot 0.6.2 运行期实际生效的 `step._tensor_stats` 与 `norm_map`。 | 扩展 `plain_config()` 兼容上述对象；`processors()` 同时采集 `stats`、`_tensor_stats`、`norm_map`、`features` 与 `device`。 |
| **P0-A4** | `scripts/paired_boundary_pilot.py:119-140, 356` | 原脚本仅观测 `preprocessor` 的 `observation.state`，完全未观测 `postprocessor`（`action` 经 `unnormalizer_processor` → `rosetta_pi_aloha_postprocessor` 的逐步变换），也未统计 `bounded_sine` 夹爪是否越界折叠。 | 新增通用 `observe_pipeline_key()` 同时覆盖预处理（`observation.state`）与后处理（`action`）两个管线，并增加 `bounded_sine_fold_report()` 实时统计 `abs(x) > π/2` 的折叠计数与极值。 |
| **P0-A5** | `scripts/paired_boundary_pilot.py:298` | `torch.as_tensor(pixels)` 对 `physics.render()` 缓冲区未做拷贝，若底层渲染缓冲复用会被后续帧静默覆盖。 | 改为 `np.ascontiguousarray(pixels).copy()` 后再转张量。 |

### 2. P0-B：新建 CUDA 闭环 `actions_per_inference` + 噪声模式扫描诊断（`scripts/diagnose_smolvla_horizon_sweep.py`）

现有 `scripts/smolvla_sim_gate.py:923-1052` 中的 `execution_strategy_diagnostic` 硬编码调用 XPU 专用 `_OnlineSmolVLA`（无法在 AutoDL CUDA 上直接运行），锁定单一噪声模式，且缺少插入任务细粒度阶段里程碑与 `bounded_sine` 折叠计数。

新建零优化器闭环诊断脚本 `scripts/diagnose_smolvla_horizon_sweep.py`（SHA-256 `a5705debd43466e6567b03e6e3b9bc5a85878c065d5317139c5fc372c2fd5b3c`）及回归测试 `tests/test_smolvla_horizon_sweep_stdlib.py`：
- 支持在 CUDA / CPU 上独立 reload 任意导出 artifact，对 `actions_per_inference`（默认 `1, 5, 10, 25, 50`）与 `noise_mode`（`seeded_standard_normal`, `zeros`）做受控网格扫描（默认 10 组条件 × 5 场景 = 50 幕）。
- 严格采用**固定每步 `env.step()`、按 `actions_per_inference` 步长刷新 chunk** 的语义，记录真实策略推理调用次数（`policy_calls`）与环境步数（`env_steps`），支持 `stop_on_safety` 碰撞/越界即停。
- 输出细粒度 MuJoCo 插入里程碑（`max_reward`、是否达 `reward >= 1/2/3/4`、左/右爪触碰、抬升、插入对齐）及每步后处理前的 `bounded_sine` 夹爪折叠率，用于与 50 步开环对比定位“开环漂移 vs 夹爪折叠 vs 视角遮挡”。
- 保持惰性导入（`--check-only` 与模块 import 不加载 `torch`/`lerobot`/`mujoco`）。

---

## 二、训练程序代码 Bug 审计与已落盘修复

对 `src/rosetta_reality/vla/`、`src/rosetta_reality/vla/training/`、`scripts/train_smolvla_v2.py`、`scripts/run_smolvla_v2.py`、`scripts/run_paired_training_v2.py`、`scripts/smolvla_forward_check.py` 等核心训练链路做逐文件排查，确认并修复 4 类代码缺陷（共改动 5 个文件，+58 / -31 行）：

| 编号 | 位置 | 缺陷详情与触发路径 | 修复内容与回归覆盖 |
|---|---|---|---|
| **T1** | `src/rosetta_reality/vla/training/basin_config.py:96-114` | YAML 中的优化器/调度器数值若写成整数（如 `grad_clip_norm: 10`、`weight_decay: 0`），`settings` 中保留 `int`，而 `launch.py` 生成的 CLI 参数是浮点字符串（`10.0`、`0.0`）；`_expect_arguments()` 按 `str(value)` 严格比对（`"10" != "10.0"`），导致 `write_training_config_bundle()` 在第 1 步之前直接抛错中断。 | 对 `lr`、`betas`、`eps`、`weight_decay`、`grad_clip_norm`、`peak_lr`、`decay_lr` 统一用 `float(...)` 规范化；单测 `test_basin_config_accepts_integer_yaml_optimizer_values_against_float_cli` 覆盖。 |
| **T2** | `src/rosetta_reality/vla/training/launch.py:141-149` | `plan["training"]["policy"]["empty_cameras"]` 原先被包在 `if mode == MODE_TRAIN:` 分支内，导致 `preflight` 和 `optimizer_smoke` 阶段静默忽略计划中的相机槽位配置（而 `compile_model`/`compile_mode` 才应仅限 `train`）。在未来引入多视角（例如启用 `empty_cameras: 0` 或双/三相机）时，会导致 `smoke` 与 `train` 输入相机槽位静默分叉。 | 将 `empty_cameras` 覆盖移出 `if mode == MODE_TRAIN:`，使 `preflight`/`smoke`/`train` 相机布局一致；`compile_model`/`compile_mode` 保持 `train` 专用。单测 `test_launch_applies_empty_cameras_overlay_outside_train_only_branch` 覆盖。 |
| **T3** | `src/rosetta_reality/vla/training/features.py:767-969` | ① `CheckpointMetricSnapshotFeature.update_policy` 原本固定以 `len(captured) + 1` 作为步号索引，一旦出现续训（`resume`）或中途步号不从 1 连续递增，就会与 `save_checkpoint(step=...)` 错位触发 fail-closed 报错；② `save_checkpoint` 直接做 `checkpoint_dir / snapshot_name`，上游传入 `str` 路径时会抛 `TypeError`；③ `GradientClipDiagnosticsFeature`、`CheckpointMetricSnapshotFeature`、`CheckpointMemoryTrimFeature` 的 `restore()` 未校验当前挂载对象是否仍为自身安装的 wrapper，存在被外层 monkeypatch 误覆盖后静默还原错层的风险。 | ① 优先读取正整数 `tracker.step`，缺失时回退 `len(captured) + 1`；② 用 `Path(...)` 包装 `checkpoint_dir`；③ 在三个 feature 中记录 `_installed_*` 引用并在 `restore()` 时严格做身份校验（fail-closed）。单测 `test_checkpoint_metric_snapshot_uses_tracker_step_str_dir_and_fail_closed_restore` 覆盖。 |
| **T4** | `scripts/run_smolvla_v2.py:179`、`tests/test_smolvla_training_features.py:102` | ① `_validate_normalization()` 实际返回 3 元组 `(report_path, manifest_path, stats_path)`，但类型标注错写为 `tuple[Path, Path]`；② 工作区已在 `features.py` 注册 `paired_mixture` feature，但 `test_registry_covers_the_declared_feature_set` 期望集合尚未同步，运行完整测试套件时会直接失败。 | 修正返回类型标注为 `tuple[Path, Path, Path]`；将 `"paired_mixture"` 补入注册表声明集合。单测 `test_run_smolvla_v2_validate_normalization_annotation_is_three_tuple` 覆盖。 |

---

## 三、已审计但未静默修改的高风险数学/配置隐患（需独立立项变更）

以下 4 项问题位于受历史哈希绑定保护的文件或需要新实验单变量立项的模块中，**严禁在无授权情况下静默覆盖**，现明确列出根因与量化证据供下一步决策：

1. **H1（高危）：`bounded_sine` 编码导致 `right_gripper` 闭合端非单调折叠**
   - **代码与数据锚点**：`src/rosetta_reality/vla/processor.py:79-101` + `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/normalization/train-only-3e3c6b9d347e5e71.json`。
   - **量化证据**：`bounded_sine` 前向映射为 $x = \arcsin(2g - 1) \in [-\pi/2, +\pi/2]$，反变换为 $g = (\sin(x) + 1) / 2$ 且**未在 `sin` 前对 $x$ 做 $[-\pi/2, +\pi/2]$ 截断**。在训练统计中，`right_gripper`（索引 13）均值为 `-0.7329`、标准差为 `0.9266`，完全闭合点 $g=0$ 对应的 $x = -\pi/2 \approx -1.5708$ 距离均值仅有 **`z = -0.904 σ`**。
   - **直接后果**：流匹配预测只要向闭合方向稍微过冲超过 `0.904 σ`（例如 `z = -2.0` → `x = -2.586 rad`），经 `sin(x)` 反而会**折返张开**到 `g = 0.238`（而非闭合 `0.0`），这与此前 2445/paired 复盘中观察到的夹爪在接近目标或抓取时异常弹开/闭合时机失准高度吻合。
   - **当前对策**：P0-A v2 与 P0-B 均已内置 `bounded_sine_fold_report()`，下一次 CUDA 诊断窗口可直接实测真实折叠比例；若实测折叠显著，需在新单变量实验中验证“反变换前 `clamp(-π/2, +π/2)`”或改用单调 `logit/atanh` 边界编码。

2. **H2（中危）：`paired_mixture_runtime.py` 硬编码 5000 步与缺少 `canonical_image_scaling` 强制校验**
   - **锚点**：`src/rosetta_reality/vla/training/paired_mixture_runtime.py:24, 199` 与 `scripts/run_paired_training_v2.py:116`。
   - **影响**：`scheduler_horizon_config` 硬编码拒绝 `registered_updates != 5000 or cfg.steps not in (2, 5000)`，后续多 epoch 或更长步数配对实验若复用该入口会被直接拦截；同时 `run_paired_training_v2.py` 未强制启用 `canonical_image_scaling`，当 `D` 视图输出 `uint8` 图像时可能与浮点归一化路径出现 `uint8/255` 舍入漂移。

3. **H3（低危/文档提示）：导出 `config.json` 中 `observation.state` 历史占位维度 `[6]`**
   - **锚点**：`runs/canonical-checkpoints-received-20260914-002/step-005000/verified/pretrained_model/config.json` 与 `scripts/canonical_furnace.py:114-120`。
   - **核实结果**：该 `[6]` 继承自 `lerobot/smolvla_base` 预训练底座，在 Rosetta 运行期内会被 14 维数据集统计显式覆盖，因此**不是** Rosetta 内部运行期维度错配 bug；但若外部工具直接读 `config.json` 而不走 Rosetta reload 协议，容易产生误判。

4. **H4（防御性加固项）：`features.py` 中 4 个早期 feature 的 `restore()` 未做 wrapper 身份比对**
   - **锚点**：`TrainOnlyStatisticsFeature`、`ActionBoundaryProjectionFeature`、`FixedFrameSamplerFeature`、`VisionFrontEndUnfreezeFeature`（位于 `src/rosetta_reality/vla/training/features.py` 前半段）仍使用 `getattr(current, "_rosetta_v2_original", None)`。当前受顺序栈卸载保护未触发故障，可在后续集中重构时统一升级为身份比对模式。

---

## 四、本地验证证据与未验证边界

### 已通过的本地校验（WSL Python 3.13.5 stdlib，无 ML 依赖、无网络、无 GPU）
- `python3 -m py_compile` 覆盖全部 10 个新建/修改 `.py` 文件：**全部通过**。
- 新增与核心 P0 单测（`test_paired_boundary_pilot_v2_stdlib.py` 7 项、`test_smolvla_horizon_sweep_stdlib.py` 4 项、`test_training_bugfixes_stdlib.py` 4 项、`test_paired_boundary_pilot_stdlib.py` 13 项）：**28 / 28 通过**（2.199s）。
- 既有纯 stdlib 回归套件（`test_execution_qualification_stdlib.py`、`test_d_train_native_view_stdlib.py`、`test_paired_measurement_files_stdlib.py`、`test_q0_state_restore_stdlib.py`）：**60 / 60 通过**。
- CLI 静态契约校验：
  - `scripts/diagnose_smolvla_horizon_sweep.py --artifact . --sim-plan . --output /tmp/p0b-unused --check-only` → `{"conditions": 10, "episodes_total": 50, "model_or_physics_executed": false, "valid": true}`。
  - `scripts/paired_boundary_pilot.py --plan configs/diagnostics/paired-boundary-pilot-20261004-002.json --check-only` → `{"source_contract_verified": true, "model_or_physics_executed": false}`（证明 2026-10-04 历史证据零破坏）。

### 明确未验证部分（需 ML 容器或获批 AutoDL CUDA 窗口）
- 本地 WSL 未启用 Docker 集成且宿主机 Python 无 `yaml`/`pytest`/`torch`/`lerobot`/`mujoco`，因此 `tests/test_smolvla_training_features.py`、`tests/test_paired_mixture_stdlib.py` 等依赖 `yaml`/`pytest` 的测试套件**未能在宿主机直接运行**。
- `scripts/paired_boundary_pilot_v2.py` 与 `scripts/diagnose_smolvla_horizon_sweep.py` 尚未在真实 AutoDL RTX 4090 CUDA 环境对 2445 / canonical checkpoint 执行前向推理与 MuJoCo 闭环；`bounded_sine` 夹爪实际折叠率及最优 `actions_per_inference` 仍待实测确认。
