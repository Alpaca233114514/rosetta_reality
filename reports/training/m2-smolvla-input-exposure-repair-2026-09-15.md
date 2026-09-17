# 83/160 输入覆盖：无需改采样，补齐评估分组

## 判定

step-2500在160个主表训练输入中见过83个，符合已登记的半程训练，不是已发现
的样本丢失或采样器缺陷。该炉使用batch4、随机打乱后不重复的20,000输入序列；
2500步恰好消费前10,000个。固定160输入主表在半程预期约80个已见，实际83个；
这项简单覆盖观察没有提供偏离登记行为的证据。step-5000已消费全部20,000个输入。

依据：`scripts/canonical_furnace.py::full_schedule`、原始模型入口记录，以及
`runs/visual-research-package-20260915-001/independent-verification.json`。
该数据包和所有旧checkpoint/计划保持原样。

需要修的是评估报告的分组能力：原006虽标注每个输入是否已见，但训练split
指标仍混合已见/未见输入，不能单独给出真正的已见输入拟合误差。
只改采样以提前见完160个评估输入，会改变实验，不是这项报告缺口的修复。

## 实现

- `visual_research.py::exposure_strata`：从已经完成配对的逐episode误差中切片，
  分别输出current_seen/current_unseen；只有非空且全部已见的组才能标记为
  `training_fit_eligible=true`。原有全split指标保留，便于追溯。
- 同时输出固定step2500已见和未见两组：`seen_by_step2500`与
  `unseen_by_step2500`。两端使用相同身份列表；5000时第二组已经进入训练，
  名称表达的是固定参考点的历史身份，不能误称5000未见输入。
- 先计算原来的全部错图和train-only基线误差，再切分子群。因此frame-0的
  非自身图像池、后续循环错图、基线样本和归一化都不随子群改变。
- 空子群输出`empty_cohort_not_measured`、`metrics=null`，不以零误差冒充通过；
  单episode仍可报告其误差，但不计算跨场景方差分解。
- 采集器核验真实ingress与schedule全序列一致，保存固定参考掩码
  `input_seen_by_step2500`，并保留schedule/ingress SHA。当前无卡，不执行采集。
- 新计划`configs/vla/visual_research_20260915_008.json`要求固定参考掩码；缺失
  则分析失败。旧API可以分析未带新字段的数组，但明确表示固定子群不可用。

## 回归设计与状态

新增四个测试：

1. 已见组MSE=0、未见组MSE=100，原混合MSE=50。修复后须分别报告，不能把50
   命名为已见输入拟合误差。
2. 切片后仍使用原有图像donor池；不能因切到已见组而改变错图实验。
3. 2500→5000固定子群身份相同；5000的当前未见组为空，不能输出0分。
4. 固定2500已见集合超出当前已见集合时拒绝分析。

最终008已通过全部四阶段：旧实现反例及修复对照、**16项CPU回归**、Ruff、
原数据覆盖复核。合成反例的旧混合MSE为50，修复后分别输出已见0、未见100，
并逐字段确认旧汇总指标保持相等。这些值验证报告逻辑，不是模型实测误差。

原数组再次复核：step2500主表已见83、未见77；step5000已见160、未见0。
总验证约6秒，最大进程树RSS约45MB，无权重加载。
证据根：`runs/visual-research-received-20260915-008/verified/runs/visual-research-008`。
修复结果及本地SHA核验见`runs/visual-research-exposure-repair-20260915-001/result.json`。

最初SSH不可用时未伪报测试通过；用户恢复无卡实例后完成动态验证。007保留
16项测试通过但Ruff因无用导入失败的记录；008移除该导入并重新验证全部阶段。
早期静态记录保留在`static-verification.json`，其中runtime_pending是当时状态。

本轮不修改训练采样、optimizer、checkpoint、原始数据或Gate。GPU用时0。
本修复不能证明模型学习改善，也不能把历史G4 0/5定位为某个唯一根因。
