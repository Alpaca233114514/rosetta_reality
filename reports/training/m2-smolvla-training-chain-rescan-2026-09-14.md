# 数据与训练链复查 · 2026-09-14

结论：本次扫描完成并修复六类配置/执行不一致缺口，16 个反例均先复现、后通过。
当前已核验的非隐藏数据没有发现数值帧错位、动作块跨轨迹或归一化统计错误。
**不能据此声明整个板块零缺陷或模型任务成功。** 最终相关回归为 512 passed、
8 failed、1 skipped、2 deselected；八个失败均是既有历史源码 SHA 不匹配，未绕过。

本报告接续 `m2-smolvla-training-chain-local-audit-2026-09-14.md` 的暂停记录。
保留原工作区修改、历史计划/源码 pin、数据、权重及失败证据；未提交或推送。
使用现有 Linux Docker，经 WSL Bash 启动；离线 CPU，内存及 memory+swap 均为 6 GiB、
CPU quota 2。没有下载、AutoDL、正式模型训练或新 Gate。合成微型策略测试执行了优化器更新，
不得将“正式模型零更新”写成“所有测试零 optimizer”。

## 确认并修复的问题

| 类别 | 原行为与影响 | 现行为 | 反例数 |
|---|---|---|---:|
| 动作边界漏装 | 删除 `action_boundary_projection` 后 schema 仍接受；实际 processor 的坐标转换/限幅由该 feature 安装，可能用原始动作配变换后统计 | v2 plan 必须声明动作边界 | 1 |
| 学习合同未执行 | 仅声明 horizon/jitter/dropout/unfreeze/K 正则合同，未声明对应 feature，仍能通过结构检查 | 合同必须绑定相应 feature | 5 |
| 参数覆盖被忽略 | smoke 自己的 optimizer/scheduler/policy，以及 `training.policy` 中的 chunk/history/action-step/video-backend 字段被接受，但 CLI 不使用它们 | 明确拒绝未实现覆盖；保持原已支持参数的行为 | 7 |
| episode 被截断 | smoke episode 为 `49.8` 时结构检查通过，下游 `int()` 可将其当作 49 | preflight/smoke 校验整数、非重复 episode 及批量/步数/运行名等字段 | 1 |
| checkpoint 声明失效 | 已注册保存网格，但允许 `save_checkpoint=false` | 正式训练必须保存注册 checkpoint | 1 |
| 草稿采样表可安装 | `draft_not_launchable`、`training_authorized=false` 的表仍可被 temporal feature 安装 | 只接受已登记且授权的表；现存草稿保持不可启动 | 1 |

反例证据位于 `runs/training-chain-rescan-20260914-001/`：
`counterexamples-before.xml`（11 个失败）、`integration-first.xml`（草稿反例失败，
其余 15 个通过）、`policy-overlay-before.xml`（4 个失败）。最终
`regression-sealed.xml` 中上述 16 个反例全部通过。
这些反例证明代码存在缺口，不证明历史 Iris/Hestia 实际触发了每个缺口。

## 检查范围与实测证据

| 路径 | 已核验内容 | 结果/限制 |
|---|---|---|
| `data/cache_resolver.py`、manifest/config | revision、episode、camera、字段映射及缓存 checksum；隐藏行在 Arrow 扫描前过滤 | 固定本地缓存通过；不下载补缺 |
| `data/dataset.py`、LeRobot adapter/native dataset | 通用 chunk anchor 的 episode/连续帧约束；SmolVLA 原生 episode→绝对/相对索引 | 两条路径区分审查；通用 action-head 路径不是当前 SmolVLA 训练循环 |
| `audit_smolvla_dataset.py` | 40 train + 5 dev，索引唯一连续、50 Hz timestamp、14-D finite、合同越界容忍 | 20,000 train + 2,500 dev 数值帧通过；隐藏样本物化 0 |
| 同上，真实图像与动作块 | 每轨迹 9 个时段，共 405 样本，state、指令、图像 uint8/CHW、全部 50×14 动作值、尾部复制及 padding | 283,500 个 chunk 动作标量与独立原始数值扫描精确一致；不再只比动作首项 |
| `audit_smolvla_processors.py` | 全训练集重新变换、计算 mean/std，并对照两臂保存的四个 normalizer/unnormalizer 文件 | 统计最大差约 4.79e-14；保存数组精确一致 |
| `audit_smolvla_saved_chain.py` | 实际保存的 tokenizer、rename、batch、projection、坐标变换、normalizer、postprocessor | 两臂各 2 episodes×3 时段；归一化与独立算式误差 0，图像与 padding 保持，动作往返最大差 1.1921e-7 |
| 固定/显式采样器、`observation.py` | 实际安装 feature→native sampler→DataLoader→Accelerate→原生 update→observer→restore | batch 1/4 × workers 0/2 四个集成测试通过；首中尾样本和掩码进入微型策略，未被首帧替代 |
| v2 plan/launch/features | 主训练与 smoke 使用的配置入口、feature 合同、恢复/累积拒绝、失败清理、源码身份 | 既有修复回归及本次新增约束通过；未来运行需新计划和源码密封 |
| loss、optimizer、scheduler、checkpoint | 原生更新路径、选中且非 padding 的 loss 分母、梯度裁剪、清零、LR、tiny 保存/恢复 | 相关合成回归通过；不等于正式 SmolVLA resume 通过 |
| 历史 selector/export | 平局分支和 reload 证据含义 | 三个历史 selector 用负 step，实际偏向较晚 checkpoint；历史七指标相等不足以证明全张量相等，仍保留为未修复历史限制 |

当前记录再次确认：首帧控制每臂 5,120 次暴露只有 40 个唯一输入帧，
对应 2,000 个唯一目标帧；新增跨时段草稿为 5,120 个唯一输入帧。
暴露次数不能写成完整数据遍历，跨时段覆盖也不等于每帧都训练过。
非隐藏原始动作有 6,973 个元素需要合同限幅，均在原注册容忍范围内；保留该事实。

## 最终验证与未覆盖部分

- `regression-sealed.xml`：512 passed、8 failed、1 skipped、2 deselected。
  8 个历史失败涉及 Zen（5）、vfunfreeze（1）、vcdropout Gate wrapper（2），
  都停在 `scripts/run_smolvla_v2.py` 的旧实现 SHA 校验；不会通过修改旧 hash 来消除。
  CUDA normalization 测试因本地无 CUDA 跳过，两项 data 测试从大回归中排除。
- `data-test.xml`：另行限定训练集的 `test_smolvla_visual_grounding_data.py`，1 passed。
- 最终 8 个本轮变更 Python 文件 Ruff 与格式检查通过，`git diff --check` 通过。
  两处旧测试 fixture 按新约束补齐动作边界、清除未启用合同，没有削弱负例断言。
- 完整真实 SmolVLA forward/backward/reload、正式恢复、CUDA 等价、新 Gate 均未执行。
- 405 张图像被解码并与样本身份一起检查；未逐帧解码全部 22,500 张图像，
  也没有独立证明视频 PTS 与物理画面语义逐帧对应。不能把数值/解码通过写成全量图像对齐证明。
- temporal feature 的原生采样/更新集成通过；两臂正式可启动训练计划、完整炉次生命周期
  和真实模型效果仍未生成/验收。历史 selector 只审查和保留，后续应使用新版本入口修复。
- Iris 当前已有 Gate 3 两臂通过、Gate 4 两臂各 0/5，M2 未完成；本审计不改变该结论。

## 证据入口

- `runs/training-chain-audit-20260914-002/result.json`、`decoded-samples.json`
- `runs/training-chain-processors-20260914-002/result.json`
- `runs/training-chain-saved-chain-20260914-001/result.json`
- `runs/training-chain-rescan-20260914-001/` 中反例、回归、环境与 data-test 记录
- 同名 JSON 保存本轮源码 SHA、精确计数及明确限制。
