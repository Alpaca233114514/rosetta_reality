# SmolVLA 数据优先科研诊断入口

本入口采集可复算证据，不执行训练、自动选 checkpoint 或自动启动 Gate。
2026-09-16已按新授权完成step5000视觉配对并确认实例关机，见
`reports/training/m2-smolvla-visual-paired-result-2026-09-16.md`。随后已补采step2500，
固定子群过程比较及再次关机记录见
`reports/training/m2-smolvla-visual-process-result-2026-09-16.md`；主对象仍为step5000。
先前无卡阶段和数据质量证据见
`reports/training/m2-smolvla-visual-research-result-2026-09-15.md`。

输入曝光分组修订008见
`reports/training/m2-smolvla-input-exposure-repair-2026-09-15.md`：半程83/160符合
原始采样计划；新分析区分当前已见/未见，并固定step2500子群用于过程比较。
修订不改变已封存的006数据包；当时封存的源码使用008计划身份，16项CPU回归与
Ruff通过。后续CUDA采集使用独立的GPU001计划及源码身份，不混用历史计划。

## 使用方式

统一入口为 `scripts/diagnose_smolvla_visual_research.py`，必需参数
`--plan`、`--stage`、`--output`；`supervision` 和 `analyze` 另需 `--input`。

| stage | 输入与输出 | 是否加载权重 |
| --- | --- | --- |
| evidence | 源码索引、旧证据 SHA 和运行环境记录 | 否 |
| data | 非隐藏数值数据、原始/投影标签、chunk/mask、逐行来源与实际训练曝光 | 否 |
| pts | 视频 packet PTS、episode 时间区间与 50 Hz 格点 | 否；不解码像素 |
| pixels | 135 个预登记帧的独立 PyAV seek 与原生解码哈希对照 | 否 |
| supervision | 固定二维坐标 3NN、训练集均值/中位数、逐 episode 指标 | 否 |
| collect | 同权重、state、语言、完整 noise 下的图像配对与完整预测数组 | 是；要求单独GPU计划和有效窗口 |
| analyze | 从完整采集数组复算指标，不调用 policy | 否 |

示例（在已登记 Linux 环境，从已回传数组复算，输出必须不存在）：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python scripts/diagnose_smolvla_visual_research.py \
  --plan configs/vla/visual_research_gpu_20260916_001.json \
  --stage analyze \
  --input runs/visual-research-gpu-received-20260916-001/verified/runs/visual-research-gpu-20260916-001/collect \
  --output runs/visual-research-reanalysis-new
```

本地数据/ML 检查须从 WSL Bash 启动 Linux Docker；远端 AutoDL 是登记例外，
使用 `scripts/run_autodl.sh shell`，不嵌套 Docker。先前无卡阶段仅运行CPU，
网络离线、两线程、RSS上限1.5 GiB；本次获批CUDA阶段使用GPU001的8 GiB RSS
上限及独立预算窗口。该窗口已结束，实例已关机；数组复算无需GPU。
输出及失败文件创建后不覆盖。

计划的 `sources` 与 `evidence_sha256` 必须匹配。历史计划 001–005 保留：
002 是完整五阶段 CPU 证据身份；004 是逐行来源/曝光修订数据身份；005 的补充
复算因输入路径错误在读取前失败；006 封存最终源码并完成最终独立复算。
006 不改变已有数据身份。重放历史计划须使用对应封存源码；不能修改历史SHA
来使当前源码通过校验。GPU001的授权窗口不能复用来启动新的采集。

## 数据包合同

交付根为 `runs/visual-research-package-20260915-001`。最外层 `manifest.json`
封存每个文件的大小和 SHA256；各阶段还有原始 manifest。数值 NPZ 不含 object
数组，使用 `numpy.load(..., allow_pickle=False)`。所有行可通过 episode/frame
关联；数组顺序以 `identities` 为准，不能假定 episode 是升序。

### data/arrays.npz 与 data/samples.jsonl

| 字段 | 轴/类型 | 语义 |
| --- | --- | --- |
| identities | `[22500,2]` int64 | episode、零起点帧；train40 后 dev5，每集 500 帧 |
| states | `[22500,14]` float64 | 原始 state，按 Action Contract 的维度顺序 |
| raw_actions / projected_actions | `[22500,14]` float64 | 原始动作与合同限幅后动作，原始值不改写 |
| timestamps | `[22500]` float64 | episode 内秒数，50 Hz |
| lower_bounds / upper_bounds | `[14]` | 合同每维边界 |
| sample_identities | `[270,2]` int64 | 每集 0/125/250/375/449/499 |
| sample_targets | `[270,50,14]` float64 | 当前帧起点，末帧重复填充 |
| sample_states | `[270,14]` float64 | chunk 起点 state |
| sample_valid_mask | `[270,50]` bool | True 才是有效监督；不能把 padding 当目标 |
| checkpoint_steps | `[2]` int64 | 2500、5000 |
| input_seen_at_checkpoint | `[2,22500]` bool | 实际消费 schedule 是否包含该输入帧 |
| valid_target_supervision_count_at_checkpoint | `[2,22500]` int64 | 不含 padding 的目标曝光次数 |

动作左右关节单位为 rad；夹爪为合同定义的归一化命令。准确的名称、顺序、
absolute/control/fps/execution 语义由包内 `contracts/aloha_insertion_smolvla.yaml`
定义；不得把原始 state 与模型标准化 state 混为一谈。

每条 JSONL 包括 `episode`、`frame`、`row_index`、`split`、`timestamp`、准确
`source_parquet` 与 SHA、dataset revision、`projection_changed_dimensions`，
以及两个 checkpoint 的输入已见标记与目标监督次数。视频文件与时间区间用
`pts/result.json.records` 按 episode 关联。原始 chunk 可从 raw_actions 按该
episode/frame 和末帧填充规则复算，投影差值为 projected_actions − raw_actions。

投影变化行是检查线索，不是已确认错标。`review-candidates.jsonl` 完整列出这些
行及原因，未按误差筛除任何 episode。物理错位/错误标签当前没有确认样本。

### 其他数组

- `pixels/images.npz`：135 个命名为 `episode_XX_frame_YYY` 的 uint8 CHW RGB，
  每张 `[3,480,640]`；对应请求时间、解码时间、旧/新 SHA 在 result.json。
- `supervision/arrays.npz`：45 个 episode 的初始二维坐标、距离矩阵、邻居行号，
  完整 500×14 原始/投影标签、3NN 预测、train-only mean/median。坐标只是初始
  两物体位置代理。参数固定为 `[640,480,640,480]` 缩放、k=3；训练查询排除自身。
- `supervision/result.json`：64 个 split×frame×window×group 记录；每个包括
  逐 episode MAE、等权均值、留一 episode 敏感性；不把时间点视为独立场景。
- `evidence/decoded-samples.json`：复用经身份核验的 22,500 个原生 RGB 哈希，
  本轮没有重复全量解码。`source-evidence/` 保留消费 schedule/ingress 证据。

### CUDA 配对数组（step5000已完成实测）

固定 train40/dev5、四个主时间点、零噪声及三个固定种子。`correct`、`wrong`
为 `[4,180,50,14]`；同时保留归一化输出、解码未限幅输出、raw/projected/
normalized targets、完整 `[4,1,50,32]` noise、mask、身份、循环 donor 和图像 SHA。
32 为当前 pinned policy 的 noise 宽度，代码从 policy config 读取，不由14维动作截断。

frame-0 必须先证明全部非视觉输入相同；其 wrong 占位数组等于 correct，分析端
明确使用同 split 所有非自身场景输出重配对。后续帧只替换循环 donor 的图像。
后续错图误差用于视觉依赖诊断，不能当作真实反事实动作标签。

每 checkpoint 预期 720 正图 + 540 后续错图 + 8 自拷贝 = 1268 forwards，
不创建 optimizer。首次自拷贝输出须精确复现；用该次耗时估计剩余采集时间，
加25%余量，不够则停止。必须另登记完整 checkpoint/processor/训练计划身份、
实际 GPU 授权、共享3600秒窗口和独立 watchdog；保留300秒回传余量。
无卡配置仍不能通过collect授权检查；GPU001计划在独立窗口下完成1268次推理。
数据位于`runs/visual-research-gpu-received-20260916-001`，新增保存的`input_state`、
`input_language_tokens`和`input_language_attention_mask`支持独立复核frame-0控制。
不可重用已过期GPU窗口或覆盖历史输出；新增采集须使用新的封存身份。

## 独立复算与解释边界

`scripts/verify_visual_research.py` 直接用保存数组做 NumPy 算术，不调用生产者的
投影、近邻或指标函数。参数为 `--input`、`--output`，可用 `--data-amendment`、
`--training-schedule`、`--training-ingress`、`--decoded-samples` 验证曝光与旧图像证据。
原始 002 整包和004修订数据的对应关系也接受检查。输出不存在才可写入。

主要模型指标为分组 `wrong MSE − correct MSE`，正负都保留；同时报告正/错图
MAE/MSE、按时点和 slot 对齐的训练均值/中位数，以及预测方差、目标方差、协方差、
均值偏差。开发集只作开发证据。checkpoint 对比必须保留固定样本并标注已见输入；
若另做子群分析，两端用同一固定子群，不能比较随 checkpoint 变化的分母。

008分析的`exposure_strata`保留`current_seen/current_unseen`及固定参考点的
`seen_by_step2500/unseen_by_step2500`，每组给出身份、样本数、当前已见数和
`training_fit_eligible`。子群切片不会重新选择错图或基线donor。空组指标为null；
固定参考掩码来自完整实际消费记录，不能仅由checkpoint名称推断。

时间格点与像素一致不证明物理视觉—state—action 语义正确；3NN 失败不证明标签
噪声不可消除；旧 CUDA parity 在其测试点通过不证明全链无错；G4 0/5 不定位根因。
