# 三臂真实输入与计划监督曝光核验

本报告核验真实 Q2 配对源进入原生处理器的工程路径，以及已封存 5000 更新计划的监督曝光；正式训练与闭环任务结果须另行读取实际输出。处理器输入通过不代表任务成功。

## 真实首批输入

AutoDL CPU helper 实际用 B/C 每臂首批 `3D+1aux`，读取真实 Q2 NPZ 和连续 50 个动作标签，复用保存的原生 CPU processor。原始动作与状态、真实 uint8 相机展开，以及 action/state/camera 归一化的独立手算均与实际处理结果逐位一致。处理器输出 action `[4,50,14]`、state `[4,14]`、camera `[4,3,480,640]`、语言 `[4,48]`。D-only stats 对象和数据身份保持。

本次耗时 8.952 秒，0 policy weights、0 forward、0 optimizer、0 physics。首个辅助样本为真实 parent 2436 的 frame 131，属于 S/R 实际共享的扰动前前缀；两臂这一输入等价是采集协议事实，不能据此声称模型学会恢复。

回传 54 个文件共 15,996,199 bytes 全部逐文件 SHA 匹配；结果源 SHA 与冻结 helper 一致。独立转存证明见 `reports/training/paired-real-CPU-input-independent-verification-20260927-001.json`。数值手算在独立 helper 中执行，本地只复核传输字节及身份，没有假称本地再次运行模型或处理器。

## 已封存计划曝光

从完整原 D train40 库存、实际 Q2 cohort 和三份 sealed 5000-update schedule 逐条重算，原 schedule entries digest 一致，三臂共同 D 输入顺序逐条相同。统计单位是有效的时间动作 slots，而非扩展 feature 维度。

| 臂 | 计划输入曝光 | 有效 target slots | padding slots | 唯一输入帧 | 重复输入曝光 |
|---|---:|---:|---:|---:|---:|
| A：4D | 20000 | 951354 | 48646 | 16227 | 3773 |
| B：3D+R | 20000 | 963303 | 36697 | 15203 | 4797 |
| C：3D+S | 20000 | 963303 | 36697 | 15203 | 4797 |

三臂共同的 15000 次 D 输入曝光含 713303 个有效 slots 和 36697 个 padding。A 的第4输入仍来自 D，保留尾部 padding：238051 有效、11949 padding。B/C 第4输入均为完整合法 50-slot 辅助 chunk：250000 有效、0 padding。因此 B/C 每臂比 A 多 11949 个有效 slots，约为 A 全臂有效监督的 1.256%。固定计算输入预算不能表述为三臂有效监督完全等量；B 与 C 则保持精确匹配。

正式配对来源为 20 个固定候选中的单个真实成功配对 parent 2436。每臂 203 个唯一辅助起点曝光 24 或 25 次，5000 次辅助输入中 3790 次来自共享扰动前历史、1210 次来自扰动后历史。它是有限单场景、成功条件下的探索性比较，不能外推为跨场景泛化或将全部 R 辅助输入称为独立恢复观测。原 legacy D 物体场景不重合仍为 `not_established`，`N_strict=0`。

完整计划计数与源身份见 `reports/training/paired-planned-exposures-independent-20260927-001.json`。以上是 sealed 计划，尚未宣称 5000 次更新实际执行；训练结束后须与 runtime 实际 batch mask 曝光计数对齐。原生 loss、尾 padding、已封存采样和动作合同均未因计数差异而调整。
