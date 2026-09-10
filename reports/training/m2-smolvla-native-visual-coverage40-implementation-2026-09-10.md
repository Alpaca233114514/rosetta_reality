# SmolVLA 40 场景视觉诊断：无卡实现模块

本轮完成评估代码、合成验证和可审阅阶段草案。**没有加载模型或真实数据，没有训练、评估或仿真；视觉泛化尚未修复，M2 未完成。** 用户授权本轮无卡 SSH、轻量实现、当前功能分支提交/推送及结束后关机；旧 30 分钟预算没有复用。

## 源码与证据

- 最终代码提交：`dfccd51f1a7117602d533a226618b8e9af06f63e`，分支 `codex/smolvla-visual-conditioning-repair-20260909`。
- 最终服务器目录相对 durable root：`dev/visual-coverage40-20260910-dfccd51`。旧 `dev/visual-utilization-20260909-001` 的 dirty tree 未覆盖，13 项 final-audit 源码哈希仍匹配。
- 从已审计归档提升了 9 个逐字节相同的源码文件，两个历史 native YAML 也保持原始字节。不是整个远端工作区备份。
- 新目录已按原始 SHA 复制 Gate 1/2、归一化小报告和历史预登记；dataset view 链接现有 durable cache，没有复制模型、数据或 checkpoint。
- GitHub 拉取发生 `GnuTLS recv error (-110): The TLS connection was non-properly terminated.`；旧远端浅仓库又缺少增量包前置提交。失败目录/记录保留，改用已推送 HEAD 的完整 Git bundle（1,856,324 bytes）和后续小增量包。所有传输都是源码/小报告。

原始结论仍见 `reports/training/m2-smolvla-native-visual-small-2026-09-09.{md,json}`：训练视觉对照 4/4，开发验证 0/4。历史 reload 只证明保存的汇总指标和首动作一致，不能补写成完整 chunk reload。

## 已实现

`src/rosetta_reality/vla/visual_coverage.py` 按冻结评估计划计算每个 view 内全部非自身错配的**平均误差**，不是先平均错配预测。每个噪声条件保存 NxN 误差矩阵、逐 episode 增益，以及全维、12 关节、2 夹爪分组结果。均值下限是有限评估集平方误差 oracle；另报 train40 均值在 dev5 上的常量基线。

`scripts/evaluate_visual_coverage.py` 接入原生 online processor 和 10 步 denoising：固定 4 组 CPU 噪声，逐图 reset；在实际 `sample_actions` 调用处记录处理后图像、状态、语言、mask、空相机占位和噪声身份。目标只参与离线计分。标准动作读取 decoder 的限幅前记录，非法输出直接失败，不让限幅掩盖误差。

证据包保存完整 50×14 normalized/standard 动作、内部夹爪、target、mask 和噪声，包含文件与数组哈希。reload 要求不同进程身份并逐值比较整个 chunk。两臂协议、非视觉条件、target 和噪声必须一致。负结果保存后退出失败；不会自动开启下一炉。失败预测留在新建 partial 文件夹。

模型执行入口要求独立封存的授权合同、最多 1800 秒的新共享 deadline、已核验外部 watchdog、资源上限和前置证据。仓库模板尚未授权，不能启动；本轮没有实现或运行 watchdog，也没有以草案替代真实预检。

## 验证

| 验证 | 结果与边界 |
|---|---|
| 恢复代码及新模块回归 | 提交 `0529143`：116 passed，15.51 秒 |
| 最终失败退出修正 | 提交 `dfccd51`：34 个重点用例 passed，1.32 秒；其中 1 项是新增用例 |
| Ruff | 初次 30 个 E501，保留失败记录；修正后通过，最终改动再次通过 |
| 四份阶段草案 | 当前状态均被真实 schema 拒绝启动；仅在内存中改变状态后结构校验通过，没有运行启动器 |
| 采样顺序 | 原生 sampler 与独立 epoch permutation 的 1024 次顺序完全相同 |
| 实际模型/数据/GPU | 未测量；实例查询返回 `No devices were found` |

两个最终测试集合合计 117 个不同用例，不是两次执行数量相加。用例覆盖均值预测与均值误差的区别、反向视觉映射、夹爪退化/失败、hidden 与非视觉漂移、padding/非 finite、证据篡改、后续动作 reload 差异和负结果退出。

采样器只使用合成 episode offsets：A 每个场景 128 次；B 中 24 个场景 26 次、16 个场景 25 次。完整顺序与 SHA 在 `reports/training/visual-coverage40-implementation-20260910/sampler.json`。真实 dataset 的索引映射仍须在优化前验证。

## 下一轮计划与启动边界

冻结依据仍是 `reports/training/m2-smolvla-native-visual-coverage40-plan-2026-09-10.{md,json}`，阈值没有改变。官方资料的核对见 `reports/training/m2-smolvla-visual-coverage-official-evidence-2026-09-10.md`：[官方文档](https://huggingface.co/docs/lerobot/smolvla)与[论文](https://arxiv.org/html/2506.01844v1)支持原生架构/训练方式；“扩大场景覆盖能改善本项目泛化”是项目假设，官方没有对此作保证。固定 256 updates、batch 4 同时改变场景覆盖与重复分配，不解释成孤立覆盖因果效应或相同 wall time。

草案目录：`configs/vla/visual-coverage40-20260910-001/`。

| 文件 | 后续阶段 |
|---|---|
| `preflight-b1.json` | batch 1、无 optimizer 前向 |
| `preflight-b4.json` | batch 4、无 optimizer 前向 |
| `smoke2.json` | 独立 fresh-base 两步 smoke，之后独立 reload |
| `main256.json` | fresh-base 40 场景、batch 4、256 步；执行 phase 仍是 `smoke` |
| `execution-contract.template.json` | 未授权合同，B checkpoint/前置证据/时间预算保留空值 |

主实验同时修改 `optimizer_smoke.episodes` 和 `fixed_frame_sampler.sample_identities`。原 `training.episodes` 已有 40 个，单改它不起作用。保持基座、processor、Action Contract、冻结范围、seed 20260809、原生 loss、optimizer/scheduler 和 bf16 accelerator 设置；不加 paired loss、dropout、unfreeze，不复用 A 或两步 smoke 的 optimizer/权重。

下一轮需先取得 GPU 与新计算预算授权，再创建**新的封存执行目录**：核验设备/环境、缓存及实际 tokenizer 依赖、磁盘预算、45 个 frame-0 的 camera/state/action/task/mask/无泄漏与实际 sample mapping。当前磁盘余量约 7.80 GiB 只是状态记录，尚未证明能容纳新 smoke/main checkpoint、原子暂存、导出与 2 GiB 余量。

然后按 doctor/benchmark → batch 1/4 前向 → 两步 smoke 与 reload → A 历史协议复现 → B 256 步 → frozen/updated tensor 与 optimizer 审计 → A/B 全量评估及各自新进程 reload → 比较的顺序执行。任何前置条件失败立即停止；不得自动重跑 Gate 1/2、删旧证据腾空间或改变学习轴。

以下是**授权、封存和前置检查完成后**的确切入口形式；`P`、`C`、`O` 指未来新登记的相对目录，当前不存在，不能直接复制草案开跑。应从已核验工作区经 `bash scripts/run_autodl.sh shell` 设置已登记平台环境，并使用 tmux/独立 deadline watchdog。不能用该脚本的历史 Way `smoke`/`formal` 包装器替代本轮原生入口。

```bash
P=runs/visual-coverage40-execution-001/plans
C=runs/visual-coverage40-execution-001/execution-contract.sealed.json
O=runs/visual-coverage40-execution-001/evaluation
python scripts/run_smolvla_v2.py preflight --plan "$P/preflight-b1.json"
python scripts/run_smolvla_v2.py preflight --plan "$P/preflight-b4.json"
python scripts/run_smolvla_v2.py smoke --plan "$P/smoke2.json"
# 必须先完成并封存 smoke reload、A 复现和资源/身份验收。
python scripts/run_smolvla_v2.py smoke --plan "$P/main256.json"
# 必须先封存 B checkpoint、冻结/更新范围与 optimizer 审计。
python scripts/evaluate_visual_coverage.py collect --contract "$C" --arm A --output "$O/A"
python scripts/evaluate_visual_coverage.py collect --contract "$C" --arm A --output "$O/A-reload"
python scripts/evaluate_visual_coverage.py reload --first "$O/A" --second "$O/A-reload" --output "$O/A-reload.json"
python scripts/evaluate_visual_coverage.py collect --contract "$C" --arm B --output "$O/B"
python scripts/evaluate_visual_coverage.py collect --contract "$C" --arm B --output "$O/B-reload"
python scripts/evaluate_visual_coverage.py reload --first "$O/B" --second "$O/B-reload" --output "$O/B-reload.json"
python scripts/evaluate_visual_coverage.py compare --first "$O/A" --second "$O/B" --output "$O/comparison.json"
```

上述入口不是完整一键编排器：smoke reload、控制复现、运行中资源/冻结审计与外部 watchdog 的执行合同仍需在新授权轮完成。封存合同须记录前置证据的实际审阅结论与哈希；不能把 `accepted` 或授权字段直接改成 true 代替验证。

验收仍要求至少 3/4 相同噪声条件下三组正确图误差均低于平均错配与均值下限，并满足训练拟合、两臂增益、物理误差不退化及完整 reload。五个 dev episodes 不是独立测试，hidden 保持封存；即使全部离线条件通过，也只能称 frame-0 开发诊断进展。

本模块在 12:20:09（Asia/Shanghai）执行官方关机，距开始约 48.94 分钟。远端关闭 SSH，随后一次连接复核也关闭；平台电源/计费状态未独立核验，实例没有释放。关机前确认没有活动任务，原审计源码和最终评估器哈希匹配，旧 checkpoint 与恢复状态文件仍在。关机及推送收尾证据见 `reports/training/m2-smolvla-native-visual-coverage40-closeout-2026-09-10.json`。
