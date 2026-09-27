# 训练与执行双时间轴诊断

本模块提供训练前后基线、逐更新观察、固定探针、异常窗口和 Basin 查询。
当前实现通过本地合成验证后才能进入真实模型准入；代码存在不代表已获训练、
模型推理或 rollout 授权。现有 Gate/M2 结论不变。

## 入口与分工

- `scripts/prepare_dual_axis.py`：只读报告与 split，创建新计划草稿和历史索引。
- `diagnostics/dual_axis.py`：版本化事件、采样表、训练侧告警、有界窗口和分块证据。
- `diagnostics/dual_axis_torch.py`、`dual_axis_bank.py`：可选 Torch 捕获、原始字节
  张量分块、固定输入封存、隔离探针、零更新端点梯度和准入比较。
- `vla/training/dual_axis.py`：最后安装的 opt-in `dual_axis_diagnostics` feature，
  包装现有 native cycle/update/clip/optimizer，不创建另一套训练循环。
- `eval/dual_axis.py`：`RolloutSink` 接入既有 `collect_reproducibility` 的可选
  `diagnostic_sink`；默认不启用，新副本保留 checkpoint/update 与 rollout_step。
- Basin：独立、安全 JSON 导入、流式校验、事件索引和确定性分析。训练器不 import Basin。

文中模块简称均位于 `src/rosetta_reality/`。ML 检查经 WSL Bash 启动 Linux Docker；
不在 Windows Python 或 WSL 主机 Python 中加载模型。

## 采集协议

`rosetta.dual_axis.v1` 使用单调 `sequence` 标识事件，`coordinates.axis` 明确区分
training/probe/execution/snapshot/history。成功 update、attempt、exposures、
sample/episode/frame、noise、rollout_step、boundary、module/call/denoise_step
分别表达不同坐标，缺失值不补零。记录 actual optimizer 成功次数；跳过更新不会推进
成功时钟，更新后失败仍保留该次已发生更新，但运行不算完整。

初始化快照在首次 native update 入口、首次 optimizer 之前生成，包含实际模型与
buffer、optimizer 参数名映射/矩状态。最终更新同样采样。完整恢复训练的 resume
不由这些诊断快照授权；导出后的独立 reload 仍使用原有正式流程。

默认固定测点为 0/1/2/4/8/16/32/64、其后每 128 次及末尾；warmup、轮次、
checkpoint 与分支边界前后各两次合并采样。重要节点使用四噪声，端点覆盖 train40/dev5；
小面板为排序后各前两个 train/dev episode 的 0/125/250/499 帧。
噪声沿用 zero、seed_20260905、seed_20260906、seed_20260907。

每次更新记录输入身份、原生 loss/已有分项、实际 noise/time、裁剪前后梯度、LR 与参数
更新抽样。完整 raw/processed batch 只在详细节点保存。采集 `embed_prefix` 的完整
调用边界，不把真实图像的 64-token 切片冒充完整前缀。原生方法记录缺失时不得声称
该边界已采到；端点 bank 的 target/mask 和 processed identity 也进入配对身份。

逐参数更新默认抽样至 256 元素；小型 norm 参数/梯度最多 4096 元素完整保留。
`sampled_not_full` 明示覆盖，完整差异由 snapshot 重算。module hook 有调用次数和元素
上限，不保留 autograd 图。窗口保留最近 16 次更新、最多 256 MiB 的序列化内部观测，
记录实际覆盖与淘汰数；触发时固化到外部文件，避免大对象进入 API。

固定探针恢复 Python/NumPy/Torch RNG、指定 generator、模型各模块 train/eval 模式、
原梯度和已登记调用状态（包括 policy queues/decoder 状态）。参数或 buffer 被改变
即拒绝，不通过回滚权重掩盖污染。真实 SmolVLA 还必须做同环境 A/A、轻量、完整采集
及后续 native 更新等价性验证。TorchLens 不在本路径中。

`seal_bank` 的 reader 必须由授权执行阶段提供，返回 canonical raw batch、合同投影后的
target 和 valid mask；它在任何读取前排除 hidden。bank 默认没有 A/A 底噪，不伪造为零。
`PreparedProbes` 使用实时 processor 核对封存输入，通过显式完整 noise 调用原生
`predict_action_chunk`。非有限值（包括 padding）和空评分 mask 拒绝。
额外端点梯度用 `endpoint_gradient_probe`，仅 train、零 optimizer/scheduler 更新。

## 分析语义

- 相对退化：同样本、噪声、输入/target/mask、processor、Action Contract、runtime
  匹配后，误差增加大于 `max(0.2 * reference, 5 * max(reference_floor, current_floor))`
  形成候选；锁定此前常规测点，下一测点再次超过才确认持续，后续恢复另行记录。
- dev 只离线分析，不改变训练侧告警/采样决策。内部统计不能单独判行为退化。
- 绝对失败：只有带阈值来源身份的有效 `absolute_limit` 才判定；可从 update 0 就失败。
- 缓慢变化另报首尾累计趋势；相邻点未越界不等于完全没有退化。
- 输出首次观测、左开右闭的可能起始区间、确认/恢复点、缺口及原始证据。
  不假设中间单调，不把稀疏区间伪报为确切首次更新。
- rollout 记录 reset、输入、processor/noise、输出、动作、下一物理状态。
  默认 reward/done/contact 变化加密；语义阶段可通过登记的 `phase_resolver` 提供，
  不把未测到阶段视为已完成。偏离后的专家时间索引动作不充当 recovery label。

## Basin 与命令

从 Basin 的 Linux 容器调用同一个 CLI/Python/MCP API：

```bash
python -m basin --store /history --source diagnostic=/evidence call basin_import_dual_axis \
  --args '{"source":"diagnostic","path":"new-bundle","run_id":"new-observation"}'
python -m basin --store /history call basin_timeline \
  --args '{"run_id":"new-observation","axis":"probe","limit":10,"field":"/metrics"}'
python -m basin --store /history call basin_locate_deviation \
  --args '{"run_id":"new-observation","section":"deviations"}'
python -m basin --store /history call basin_branch_compare \
  --args '{"left":"control","right":"treatment"}'
```

新 adapter 的事件使用独立 shards，不放入旧的单一 `events.jsonl`；请用双时间轴 API。
旧格式和旧接口保持兼容。`basin_history` 使用新记录的 event_count。
时间线分页，分析器依次读取有界 shard；二进制不反序列化 Python 对象。
`basin_read_artifact` 可用已登记 source 加 byte_offset/byte_limit 读取最多 4096 字节
外部切片，并重新流式计算整个文件 SHA。普通 history verify 只认证已导入字节，不能
认证外部张量仍然存在。当前会话是否自动热加载新增 MCP 工具需另行确认。

## 准入与验证

草稿生成：

```bash
python scripts/prepare_dual_axis.py --output /output/prepared \
  --split configs/diagnostics/dual_axis_split_20260924.json
```

此命令只准备文件。实际 feature 还需要 registered 状态、源计划授权、封存 bank/module/
动作组、当前 runtime、implementation identity、真实模型透明性准入、资源/截止边界。
仅改 draft 布尔值不足以执行。常规开销目标 15%；端点扩展及异常加密另计，超预算不
静默降密度。现有学习语义、权重选择、Gate 标准保持不变。

`scripts/run_dual_axis_tests.sh` 使用已有 digest 镜像、禁网、只读源码、4 GiB/2 CPU。
历史 fixture 测试需在普通容器目录放置独立核验的 metadata JSON，避免 Windows
junction 穿出根目录；不可放宽生产路径校验。
