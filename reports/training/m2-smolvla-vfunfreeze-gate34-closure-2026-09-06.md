# SmolVLA 视觉前端解冻候选（vfunfreeze-003）收线报告 — 2026-09-06

单轴候选 `m2-smolvla450m-vfunfreeze-003`（plan
`configs/vla/smolvla_450m_aloha_insertion_vfunfreeze_cuda_b32_003.yaml`，
登记回退 batch 32 / 632 步 + 视觉前端 per-module activation
checkpointing）按预注册链完整执行并收线。**结论：对齐门通过、offline
指标创项目最好记录，但 Gate 4 仍为 `0/5`——视觉解冻轴作为第七个
0/5 身份关闭为负结果。**

## 1. 执行链与结果

| 阶段 | 结果 | 证据（本地 runs/ 已校验回传） |
|---|---|---|
| doctor / benchmark / preflight | passed | 远端 durable run root |
| 两步 optimizer smoke + 更新验证 | **PASSED**：视觉前端 198/198 张量非零更新；语言模型 147/147 张量逐位相同 | `diagnostics/vfunfreeze-smoke-update-verification.json` |
| formal 训练（632 步，5.29 s/step） | 完成，梯度健康（pre-clip L2 1.5–1.8 < clip 10） | 远端 gradient-clip JSONL |
| 逐 checkpoint 验证 | 单调收敛 0.0407→0.0252→0.0221→**0.0196** | `selection/…-selection.json` |
| validation-only 选样 | step 632，first-action MAE **0.019603**（base 0.290538，93.25%；优于 Zen-uniform 0.021573 / firstaction 0.022150 / Aster 0.022510 / vcdropout 0.029304——跨 batch 档登记警示适用） | 同上 |
| export + 独立 reload | manifest `verified`，7 项确定性指标差 **0.0** | 远端 artifact manifest |
| 帧 0 对齐门（入场许可） | **PASSED**：零噪声增益 +0.000563>0，4 噪声设定 3 正（对照 vcdropout：变号）；image_output_shift 降至 0.0066–0.0078 | `diagnostics/frame0-paired-alignment-validation-vfunfreeze-2026-09-06.json` |
| Gate 3（seed 20260809，20 步） | **passed** | `gates/gate3-smolvla-sim-444.json` |
| **Gate 4（seeds 1000–1004，500 步）** | **failed：`0/5`，success_rate 0.0**；19 joint-limit violations、19 unexpected collisions（vcdropout 131/34，Zen-firstaction 0/0）；动作全部有限、无 invalid、无裁剪 | `gates/gate4-smolvla-sim-444.json` |

部署 artifact `m2-smolvla450m-vfunfreeze-cuda-b32-003-step0632-deploy-001`
保留在远端 durable 数据盘（含 inventory backup 证据）；1 GB 权重按流量
约定暂不回传本地，需要时单独授权。

## 2. 过程事件（全部按停止条件处理，负结果证据保留）

- 候选 `-001`（batch 64）：smoke CUDA OOM（~23.5 GiB 需求 vs 23.53 GiB）。
- 候选 `-002`（batch 32、无 checkpointing）：smoke 再次 OOM——可训练
  SigLIP 塔逐层反传激活 ~0.6–0.8 GiB/样本。
- 候选 `-003` 中途：系统盘（30G）在 checkpoint 158 写 optimizer state 时
  打满 → checkpoint 根迁数据盘后重跑（失败 formal 残骸可逆移至
  `superseded-*` 目录保留）。
- 服务器自主执行 + watchdog（成功/失败/停滞均落报告并自动关机）按设计
  工作：`runs/orchestration/vfunfreeze-watchdog-final-2026-09-06.json`
  （outcome=success，链路 09:25 UTC 完成）。

## 3. 判读

1. **表示轴的最后一个主假设被否证**：让视觉前端可训练（语言模型逐位
   冻结、更新到达经过权重级证明）确实同时改善了 offline MAE（历史最佳）
   与帧 0 图像-专家对齐（首个通过对齐门的候选），但闭环任务成功率仍为
   零。`offline 改善 → 对齐改善 → 任务成功` 的转化链在最后一环断裂。
2. 至此七个身份（Faust、Aster、Way、Zen-uniform、Zen-firstaction、
   vcdropout、vfunfreeze）在完全一致的 Gate 4 协议下全部 `0/5`。
   时间加权、状态鲁棒、状态条件化、视觉解冻四条学习侧轴全部关闭；
   失败不再能归因于"看不到图"或"视觉表示不可训练"。
3. 剩余假设族（供下一轮预注册，本报告不授权任何新炉）：闭环控制/
   动作执行层（19 joint-limit violations 提示 chunk 执行与控制边界）、
   T2 恢复监督（教师线已在协议墙收线）、数据规模/任务难度（50
   episodes / 316–632 步的 development-scale 上限）。

## 4. 证据清单

- 本地：`runs/<expid>/gates/gate{3,4}-smolvla-sim-444.json`、
  `selection/m2-smolvla450m-vfunfreeze-cuda-b32-003-selection.json`、
  `diagnostics/frame0-paired-alignment-validation-vfunfreeze-2026-09-06.json`、
  `diagnostics/vfunfreeze-smoke-update-verification.json`、
  `runs/orchestration/vfunfreeze-watchdog-final-2026-09-06.json`。
- 远端（durable 数据盘）：formal checkpoints（4）、deploy artifact +
  inventory backup、全部相位日志与事件流、渲染 sim plan
  `configs/vla/m2-smolvla450m-vfunfreeze-cuda-b32-003-sim-444.yaml`。
- 预注册与两次 OOM/回退修订：
  `reports/training/m2-smolvla-vision-front-end-unfreeze-preregistration-2026-09-05.{md,json}`。
- 代码：`src/rosetta_reality/vla/vision_front_end.py`、
  `vision_front_end_unfreeze` feature、vfunfreeze 协议/验证/选样/导出/
  smoke 验证/炉子/AutoDL gate 包装器（suffix 444）。
