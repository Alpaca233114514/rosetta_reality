# SmolVLA 视觉前端解冻候选预注册 — 2026-09-05

单轴候选 `m2-smolvla450m-vfunfreeze-001`（plan
`configs/vla/smolvla_450m_aloha_insertion_vfunfreeze_cuda_b64_001.yaml`，
hash 由文件本身固定）。本预注册在启动任何付费 CUDA 工作之前冻结假设、
身份、验收与停止条件。用户于 2026-09-05 明确授权启动本轴训练试验。

## 1. 假设与动机

六个已完成身份（Faust、Aster、Way、Zen-uniform `411`、Zen-firstaction
`422`、vcdropout `433`）在完全一致的 Gate 4 协议下全部 `0/5`。修正后的
2026-09-05 诊断（AutoDL 复测，报告在
`runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/`
下）建立了两个事实：

1. 策略输出对帧 0 图像内容敏感（`image_output_shift` 0.016–0.019），
   但正确图像相对错配图像的配对 MAE 增益仅 ±0.0001–0.0008 且跨噪声
   seed 变号——敏感存在、对齐为零；
2. 冻结 tower/connector 表示（含 2×2 空间读出）在 train→validation 纪律
   下全部无法线性解码首动作（0.0164–0.0169，劣于常量基线
   0.0150/0.0139）；上下文化 KV 未测，不宣称信息不存在。

假设：**可训练头缺乏可读的视觉表示是帧 0 对齐为零的直接原因之一；
把视觉前端（`vlm.model.vision_model` + `vlm.model.connector`）加入可训练
集、其余语义保持冻结基线不变，可以把"图像敏感"转化为"图像对齐"，并
进而改善闭环任务成功率。**

反例风险（预先登记）：若动作专家结构性地忽略视觉 KV，或 316 步不足以
重塑表示，本轴将以与 vcdropout 相同的方式失败——这是对假设的直接
检验，不是对工具链的检验。

## 2. 单轴声明

对照 Zen-uniform 控制系（fresh pinned base、uniform flow loss、batch 64、
316 次更新、20,224 exposures、同一 optimizer/scheduler/seed/split），
本炉只改变一个变量：

- **处理**：trainer 进程内 `make_policy` 返回后，把恰好
  `vlm_with_expert.vlm.model.vision_model` 与
  `vlm_with_expert.vlm.model.connector` 的参数置为可训练
  （`vision_front_end_unfreeze` feature，`src/rosetta_reality/vla/vision_front_end.py`）。
- 语言模型（text model / lm_head / embed_tokens）保持逐位冻结；
  upstream 作用域旗标与保存的 policy config 旗标保持冻结基线
  （`freeze_vision_encoder=true`、`train_expert_only=true`、
  `train_state_proj=true`），因此验证/导出/部署加载时前端自动重新冻结；
  feature 安装时强制校验父实验 adaptation 仍是该冻结基线（九月五日
  launcher 修复拒绝无效解冻组合，本设计绕开该组合而非放宽它）。
- 数据、loss、optimizer/scheduler、batch、steps、seed、split、
  normalization、监控策略全部与 Zen/vcdropout 系一致。
- 禁止共处理：`state_conditioning_dropout`、`state_robustness_jitter`、
  `horizon_weight_profile`、`fixed_frame_sampler` 均为 FORBIDDEN。

技术备注：pinned 上游没有"只解冻视觉"模式（`train_expert_only=true`
冻结整个 VLM；关闭它开放文本层），因此可训练范围必须显式声明并在
安装时逐参数核验。optimizer 由 `policy.parameters()` 构建（pinned
`get_optim_params`），无需参数组改动。SigLIP 塔与 connector 仅含
LayerNorm/GELU，`train_expert_only` 强制的 eval 模式不改变其前向数学，
梯度照常回传（expert 前向无 no_grad 段）。

## 3. 身份冻结

- 父实验 `m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003`
  （sha `0e9dd049…`）；runtime profile `autodl-rtx4090-cuda-001`
  （sha `be2bfc3e…`）；dataset/model/processor revision 与既往一致。
- 初始化：revision-pinned base；不复用任何已完成炉次的 checkpoint 或
  optimizer state。
- 训练：40 train episodes（同 Zen 序）、batch 64、316 步、
  checkpoint 79/158/237/316、AdamW lr 1e-4 / betas (0.9,0.95) /
  wd 1e-10 / clip 10、warmup 31、cosine 至 2.5e-6、seed 20260809。
- 验证：episodes [22,13,7,33,45]、frame offset 0、零噪声；hidden
  [31,6,1,24,5] 保持密封。
- 实现文件哈希见 plan `implementation_files`（15 个文件，全部
  create-only 新增或本次会话修改后立即冻结）。
- 监控：固定 `sleep 300` 五分钟阻塞，仅 quarter-checkpoint 唤醒做完整
  审计；每相位时长预算合计 150 分钟（formal 相位），超时即停止条件。

## 4. 证据链与验收（全部 create-only）

1. **阶梯前置**：doctor → benchmark → no-optimizer preflight →
   两步 optimizer smoke（与 vcdropout 炉同构，通过
   `scripts/run_smolvla_v2.py`）。
2. **smoke 更新验证（新增相位，gating）**：
   `scripts/verify_vfunfreeze_smoke_updates.py` 对照 revision-pinned
   base 快照逐张量比较两步 smoke checkpoint——
   (a) 视觉前端参数存在且 ≥50% 发生非零更新（下限登记为 0.5，实测值
   全量入报告）；(b) 语言模型参数存在且逐位相同；任一不满足则轴在
   smoke 层关闭，不得进入 formal。
3. **formal 训练** → 逐 checkpoint 验证 → validation-only 选样
   （`first_action_mae`，tie-break 早 checkpoint）→ export + 独立
   reload 精确相等（与 vcdropout 链同构的 create-only 包装：
   `smolvla_vfunfreeze_validate.py` / `select_smolvla_vfunfreeze_checkpoint.py` /
   `export_smolvla_vfunfreeze.py`）。
4. **对齐门（gating，本地）**：artifact 回传本地后，用修正探针
   `scripts/diagnose_frame0_vision_probe.py --split validation` 复测：
   零噪声 `paired_mae_gain > 0` 且四个噪声设定中 ≥3 个为正，才允许
   登记 Gate 3/4 对比；否则轴在对齐层关闭。对照基线：vcdropout
   artifact 同协议增益 ±0.0001–0.0008 且变号。
5. **Gate 3 / Gate 4（gating，本地，单独注册）**：与六身份完全一致的
   协议（seeds 1000–1004、500 步、同 Action Contract 与碰撞分类）。
   **主判据：Gate 4 成功数 > 0** 即方向性证据；`0/5` 则本轴作为负结果
   关闭，与 vcdropout 同待遇。offline MAE 与梯度比率一律不作为验收。

## 5. 停止条件

继承 vcdropout 全部停止条件，另加：
`vision_front_end_scope_or_smoke_update_verification_failure`。
OOM 不自动重试：batch 64 若在 benchmark/smoke 层 OOM，本炉停止并报告，
任何降 batch 重注册需用户显式批准（不自动降档）。

## 6. 边界与不做的事

- 不修改已完成炉次的 plan/hash/evidence；不复用其 optimizer state。
- 不触碰 hidden test；不在计费 GPU 上做实验分析（分析回本地）。
- 教师线 -004 与恢复数据轴保持独立，不因本轴结果隐式开闭。
- 本预注册不构成对视觉表示问题的终局判定；KV 层探针仍未实现。

## 7. 修订一（2026-09-05）：候选 -001 smoke OOM 与登记回退 batch 32

- 候选 `-001`（batch 64/316 步）在两步 optimizer smoke 第一步反传时
  CUDA OOM（需 ~23.5 GiB，4090D 实际 23.53 GiB；冻结基线 vcdropout 峰值
  18.29 GiB，解冻视觉前端的反传激活增加 ~5 GiB 以上）。炉子按
  `out_of_memory_no_automatic_retry` 停止，失败证据在远端
  `runs/orchestration/vfunfreeze-furnace-events.jsonl` 与
  `vfunfreeze-phase-smoke.log`；`-001` 的 plan 与预注册保留为不可变
  失败证据。
- 用户于 2026-09-05 显式批准登记回退：候选 `-002`
  （`configs/vla/smolvla_450m_aloha_insertion_vfunfreeze_cuda_b32_002.yaml`）
  ——batch 32 / 632 步，是保持 20,224 exposures 精确等量的唯一干净
  分档；warmup/decay 等比调整为 62/632；其余身份（数据、loss、
  optimizer 系数、seed、split、scope、监控、停止条件）全部不变。
- 边界：与 Zen-uniform 控制（batch 64/316 步）的对比跨 batch 档，
  与 Way 当年 128→64 回退同待遇，作为登记回退解释，不冒充同档单轴。
- 显存预估：~16 GiB 峰值（余量 ~7 GiB）；预估 formal 相位时长 180 分钟。

## 8. 修订二（2026-09-05）：候选 -002 再 OOM 与 checkpointing 注册

- 候选 `-002`（batch 32、无 checkpointing）在两步 smoke 第一步再次
  CUDA OOM（23.09 GiB）。参数范围报告（smoke 相位 create-only 证据）
  证明恰好 198 个张量新可训练、语言模型全冻结——OOM 是解冻 SigLIP 塔
  的真实激活成本：512×512 → 1024 token、隐藏 1152、~27 层，反传需
  ~0.6–0.8 GiB/样本的逐层激活（冻结时为零），batch 32 即 ~20 GiB+。
  失败证据：`vfunfreeze2-furnace-2026-09-05.log` 与事件流。
- 用户批准方案 A：视觉前端 **per-module 非重入 activation
  checkpointing**（`torch.utils.checkpoint`，`use_reentrant=False`，
  梯度关闭时纯直通；非重入实现保存/恢复 RNG，dropout 可复现）。
  单元测试证明输出与梯度与无 checkpoint 逐位相等（float64 含
  dropout）。
- 候选 `-003` = batch 32 / 632 步（修订一回退形状）+ checkpointing。
  batch 64 即使 checkpointing 预估 ~27 GiB 仍不可行；batch 32 +
  checkpointing 预估 ~18–21 GiB。合同新增
  `activation_checkpointing: per_module_nonreentrant_vision_front_end`，
  checkpointing 由 feature 在 trainer 进程内安装，验证/导出/部署路径
  不经过 checkpoint 分支。
