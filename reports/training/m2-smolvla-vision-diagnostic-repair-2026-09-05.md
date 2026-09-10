# SmolVLA 视觉诊断修复 — 2026-09-05

本次修复诊断工具与无效训练配置检查；没有运行真实模型、读取真实数据行、
改变权重、启动训练、连接 AutoDL 或修改历史 Gate。M2 仍未通过验收。

## 修正的证据边界

9 月 3 日的报告和 JSON 保留原样，但以下推论不能继续作为当前事实：

- 平均池化后的线性回归失败，只说明该读出方式在该样本规模下未成功，不能推出
  原始 token 不含物体位置、动作专家没有可学习的信息或必须解冻视觉塔。
- connector 输出还会经过 VLM 的文本层；它不等于动作专家实际注意的上下文化 KV。
  本次空间探针仍只覆盖 tower/connector，KV 层诊断没有实现，不能声称已覆盖全链路。
- 输出随图像变化，与输出朝专家动作的正确方向变化，是两个不同的指标。
  首动作也可能包含示范者偏好；与单条专家动作不一致不自动等于任务不可成功。
- 0.0134 是历史常量预测参考，不是动作误差或任务成功的理论下限。
- 原脚本先 materialize 全部首帧，再排除 hidden action；spread 脚本还在所有首帧
  state 上计算差异。这与旧报告 `hidden_test_loaded=false` 的文字边界不一致。
  代码审计未证明 hidden 参与了回归拟合，也未复查旧运行时日志。

## 已实现

共享模块 `src/rosetta_reality/vla/vision_diagnostics.py`：

1. 从 checksum-bound 父实验读取 train/validation/hidden split，验证集合互斥。
2. 标签和图像共用同一个 `resolve_prepared_cache(..., validate_checksums=True)`
   解析结果；拒绝实验与数据配置 revision 不一致。
3. Arrow 扫描同时限制 `frame_index == 0` 和明确的 episode allowlist，随后才
   materialize 行；拒绝重复、缺失、非 finite 样本。校验共享文件字节不等于
   加载其中 hidden 样本，本次隔离保证的是禁止 materialize hidden 行。
4. 校验 deploy manifest、文件 SHA-256、reload 声明和 train-only normalization。
5. 双空间读出：全局平均、保持网格位置的 2×2 池化。只在 train 内部五折选择
   ridge alpha；fit/标准化均不使用 validation。使用 dual ridge 避免构造巨大
   feature-by-feature 矩阵。训练 CV 是调参分数，validation 才是独立读出评分。
6. 同时报告 train mean/median 常量基线、逐维误差；验证集不选择读出方式。

入口变化：

- `scripts/diagnose_first_action_spread.py`：默认只读取 40 个 train episode。
- `scripts/diagnose_frame0_vision_probe.py`：默认 train，可显式选择 validation；
  对严格相同的首帧 state，固定 instruction 和 noise，对比正确图像与循环错配
  图像。报告 zero noise 及固定 seeds 20260905/20260906/20260907 的结果，
  保存逐 episode 配对 MAE 改善、逐维相关性和输出变化，全部 `gating=false`。
- `scripts/diagnose_frozen_feature_probe.py`：40 train 拟合，5 validation 独立评分；
  显式 `--tower-grid H W --connector-grid H W` 必须与运行时 patch size、
  pixel shuffle scale 和 token 数量一致。只报告结果，不输出“信息不存在”判决。
- 两个模型诊断入口都支持 `--output` 写入新的 JSON 文件；已有文件拒绝覆盖。
  报告保留 artifact manifest hash、dataset revision、episode/noise/readout 身份。

`src/rosetta_reality/vla/training/launch.py` 在生成命令前拒绝：

- `freeze_vision_encoder=false` 与 `train_expert_only=true` 的无效解冻组合；
- 非布尔 adaptation 值；
- 在正式计划或 phase/policy overlay 中添加却不会生效的 adaptation 覆盖字段。

合法冻结基线的 CLI 不变。真正的视觉适配需要新的 checksum-bound 父实验，
不能修改已完成炉次的 plan/hash。也不能直接把两个开关都关掉后声称“只训练视觉”：
这还可能开放文本层，必须单独声明可训练参数范围并核验实际梯度/更新。
旧 plan 的 implementation checksum 保留；本次改变 launcher 后，新的运行必须
重新登记实现身份，不能用旧 plan 绕过 hash 漂移检查。

## 验证状态

- 已新增 synthetic 测试：scan-time hidden 隔离、重复/缺失样本、空间池化保留位置、
  validation 标签不能影响 alpha、dual/primal ridge 一致、敏感但错误的动作案例、
  常量维度相关性、单一 cache identity，以及无效解冻/静默覆盖的拒绝路径。
- 本次修改文件的 Ruff 静态检查及 `git diff --check` 通过。
- 容器测试尚未执行；新增测试用例不构成通过证据。
- Docker Desktop 状态检查曾返回 stopped，启动命令返回 already running，
  Linux engine `_ping` 连续返回 HTTP 500；没有重启 Docker、WSL 或其他任务。
- 真实模型复测、KV 探针、梯度与权重更新验证、小样本对照、Gate 3/4 均未执行。

容器就绪后的无权重/无数据测试命令（WSL Bash）：

```bash
bash scripts/run_m2_container.sh vla-cpu python -m pytest -q \
  tests/test_smolvla_vision_diagnostics.py \
  tests/test_smolvla_training_launch.py
```

## 下一阶段

先完成修复工具的容器测试与诊断复测，再根据空间读出/实际 KV 的证据决定修改
视觉表示还是动作专家的信息利用。视觉适配仍属于实验候选；必须先通过明确
参数名单、optimizer 覆盖、有限梯度、非零更新与独立 reload 的 tiny smoke。
正式比较需要冻结基线与候选在同一数据、batch、loss、更新次数下运行。
视觉对齐改善不能替代 Gate 3/4，恢复数据仍是独立轴。

## 验证补记（2026-09-05，容器测试与诊断复测完成）

- 本地 Docker engine 恢复后，`vla-cpu` 容器执行
  `tests/test_smolvla_vision_diagnostics.py` 与
  `tests/test_smolvla_training_launch.py`：28 passed；Ruff 对本次全部
  修改文件通过。AutoDL（RTX 4090D，workspace
  `20260905T031903Z-2078b642dd96-4c6874438098`，archive SHA-256
  `4c68744380987403c96a1ca77de06add1e20cf19b2dd0fb162fc944dbf2c6fc4`）
  doctor 通过后在同一 workspace 复跑：同样 28 passed + Ruff 通过。
- 修正后诊断在 AutoDL tmux 内对默认 artifact
  `m2-smolvla450m-vcdropout-cuda-b64-001-step0237-deploy-001` 复测，
  报告写入远端 durable runs 后回传本地（SHA-256 一致）：
  - 配对探针（train 40 / validation 5，噪声 zero+3 固定 seed）：
    `image_output_shift` 0.016–0.019，但配对 MAE 增益仅
    ±0.0001–0.0008 且跨 seed 变号——图像敏感存在，帧 0 对齐为零；
  - 空间读出探针（tower 32×32 / connector 8×8，train 内选 alpha、
    validation 独立评分、hidden 行 scan 期排除）：tower/connector ×
    mean/2×2 四种读出 validation MAE 0.0164–0.0169，均劣于 train 均值
    基线 0.0150 与中位数基线 0.0139；
  - 首动作 spread（train-only）：MAD 0.0132，成对 0.0182，frame-0
    状态逐位相同（0.000000）再次确认。
- 结论边界：九三报告的方向性读数在修正协议下存活（池化与 2×2 空间
  线性读出均失败；敏感≠对齐），但上下文化 KV 仍未测，不能据此宣称
  视觉信息不存在。诊断证据文件位于
  `runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/`
  （`frame0-paired-alignment-{train,validation}-2026-09-05.json`、
  `frozen-spatial-readout-2026-09-05.json`、`visiondiag-2026-09-05.log`）。

