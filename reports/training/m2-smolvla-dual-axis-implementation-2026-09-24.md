# Rosetta × Basin 双时间轴诊断实现与验证

已实现工具、派生历史索引和未来实验准备。本轮没有加载真实模型/数据、训练、
rollout、SSH、提交或推送；没有新增 Gate 结果，M2 仍未完成。

## 已交付

- 原生 v2 `dual_axis_diagnostics` feature：实际初始化后的 update-0 快照、逐更新
  loss/梯度/抽样参数更新/实际 noise-time 观察、固定输入探针、最终快照。
  成功更新、尝试更新、完成样本曝光分开；跳过/中途失败保留真实状态。
- 默认加密采样表、四噪声 panel、16 次更新/256 MiB 触发前窗口、完整小 norm
  数组和有界 module/prefix/denoiser 观察。raw 明确指 processor 前交付 batch，
  不冒充原始磁盘视频字节。大数组独立分块，不嵌入 MCP 回包。
- 输入 bank 封存/校验、隔离探针、train-only 零更新端点梯度 helper、注册/源码
  闭包/runtime/资源/截止拒绝门禁。梯度及矩状态按真实参数名关联。
- 既有配对 rollout collector 的可选 `RolloutSink`：关联 checkpoint/update 与
  reset、每步完整输入/noise/动作、下一物理状态和接触变化；默认未启用。
- Basin 三个只读 API：`basin_timeline`、`basin_locate_deviation`、
  `basin_branch_compare`。CLI/Python/MCP 共用实现；当前目录有 12 个默认只读工具。
  新 bundle 流式核验并保存 shards，保留旧 adapter；外部张量按受控 source alias
  校验并读取有界切片。
- 历史索引覆盖 12 个主要分支，另有 4 个优先窗口，共 16 条派生事件、21 个文件；
  保留报告/机器读数来源哈希，研究导航关系不等价于权重继承或因果链。

使用说明与接口合同：`docs/dual-axis-diagnostics.md`。Basin 对应说明是其仓库中的
`docs/dual-axis.md`。两个仓库均留在原有功能分支；Basin 写入前核对源/目标 SHA，
6 个改动文件均已落地，原有文件另有 create-only 备份，旧 history 未修改。

## 准备产物

最新草稿：`.cache/dual-axis-prepared-002/prepared/plan.json`。
历史索引：同目录 `history/`。草稿为 32 次更新、warmup=16，实际合并节点为
0/1/2/4/8/14/15/16/17/18/32；小面板 16 个样本，端点面板 180 个 train/dev 样本。
13 个实现文件的封存 SHA 已重新核对当前源码。

草稿不可调度：真实输入 bank、精确 module/action 语义、真实模型 A/A/light/full
透明性与后续更新等价性、实测资源/吞吐和截止仍须在独立授权阶段完成。
不能只改 status/授权布尔值进入执行；也不能用合成通过替代这些证据。

## 实际验证

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 环境 | WSL → 既有固定 Linux Docker，禁网、只读源码、2 CPU/4 GiB；Torch 2.11.0+xpu，CUDA/XPU 设备均不可用 | `.cache/dual-axis-final-regression-006/container.log` |
| 相关完整回归 | 136 passed | 同上 |
| 最后运行边界复测 | 13 passed，包含后来新增的 bank → 64 个固定噪声探针与 tail-mask 验证；与前述计数存在重叠，不相加 | `.cache/dual-axis-bank-final-008/container.log` |
| 静态检查 | Ruff passed；两个仓库 diff whitespace 检查通过 | `.cache/dual-axis-final-lint-004/container.log` |
| 实际 Basin CLI/MCP | 12 个只读工具；7 条协议响应，查询/比较/校验通过 | `.cache/dual-axis-api-demo-002/mcp-protocol.json` |
| 退化定位反例 | `(64,128]` 区间，129 确认、130 恢复；保留原始引用 | `.cache/dual-axis-api-demo-002/verification.json` |

合成原生训练比较核对最终参数、梯度、loss、optimizer/scheduler 状态、Python/NumPy/
Torch RNG 与后续更新。fake-environment rollout 核对插入 sink 前后执行动作逐值相同。
另外覆盖首步/中途异常、瞬态恢复、累计缓慢变化、身份/noise 不匹配、缺失 A/A 底噪、
内部变化不冒充行为退化、跳过更新、更新后失败、重复模块调用、窗口淘汰、篡改/中断、
外部切片哈希、分页、隐藏集读取前拒绝及草稿/源闭包拒绝。

两条测试 warning 分别是容器没有 XPU 和测试夹具为了独立捕获梯度包裹 optimizer.step；
实际检查同时比较了 scheduler/optimizer 状态，不把 warning 当作生产精度验证。

早期失败目录均保留：原生测试 fixture 缺少 `update_metrics`、临时 Basin checkout
漏拷 MCP 启动脚本、只读 Ruff cache、Windows runs junction 导致历史 fixture 缺失。
均通过修正测试准备解决，没有降低生产安全校验。历史 config fixture 的 SHA 为
`978ec7b1b9b648749b96cf0967ae4b37e0133d65e491613e6803884d50c0984a`。

## 证据边界

真实 SmolVLA/CUDA 插桩透明性、端点梯度、独立模型 reload 和 15% 常规开销目标均
尚未测量。本轮的查询示例是明确构造的合成异常，不是发现了真实训练的首次退化。
历史数据不足时只返回观测区间/缺口，不重建不存在的梯度或轨迹。
额外端点和异常窗口成本另计，实测超预算就停止扩展，不自动降采样或扩大资源。
当前 Codex 会话是否热加载新增 MCP 工具未验证；新进程的 CLI/MCP 已实际验证。
