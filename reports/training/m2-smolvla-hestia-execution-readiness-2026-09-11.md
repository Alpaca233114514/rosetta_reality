# Hestia下一执行阶段：已准备，等待新增GPU预算授权

本次只读SSH与19文件取回已结束，实际B的偏差、时移和动作分量诊断已完成。
偏差/时移均0/4；夹爪场景响应在训练集已经压缩，开发左夹爪相关为负。
这些结果支持下一步区分拟合不足与跨场景映射失败，不能靠继续后处理宣称修复。

## 待批准的具体动作

沿用 `m2-smolvla-hestia-main-plan-2026-09-10.md` 的唯一已登记候选
`m2-smolvla450m-visual-hestia-fit40-001`，不另开实验轴：40个frame-0场景、
batch4、1280 updates，warmup16、cosine decay1280、peak LR1e-4、末端2.5e-6。
这是更新预算和LR轨迹的联合干预，不是纯重复次数效应。B原件保持只读。

计算supervisor上限1800秒，含当前CPU/真实前置验收、训练、封存、B/C四次
完整采集与独立reload；另有120秒证据回收窗口及既定关机流程。
完成或失败按已登记检查执行官方关机，不释放实例，不自动重试或下一炉。
当前实例需有计划要求的单张4090 D；无卡实例只支持已完成的文件取回，
不满足此GPU执行前提。新预算和执行后关机必须由用户明确授权。

## 已完成与仍待现场核验

本地源码 `11d2e2b` 包含197项Hestia合成回归、精确5120合成样本顺序及
AST不变的格式修正；详细结果见 `m2-smolvla-hestia-local-regression-2026-09-11`。
模型/数据没有因本地检查而被重新加载。源码准备不等于当前CUDA平台通过。

批准后用内容定址Git bundle在新版本目录创建独立detached checkout，
不对旧dirty workspace执行pull或覆盖；不推送GitHub。bundle基于已现场
核验的Hermes commit `02cbf63269c2f3038e9e2cb7ec1af64d84b17813`。
仅传源码；模型/数据缓存只读复用，经manifest/revision检查，不新增下载。

必须现场重验实例/GPU、源码/依赖、B和旧smoke checkpoint/processor/config、
历史B全数组、数据view、空闲磁盘、其他活动任务及supervisor身份/期限。
候选输出已存在、剩余空间不足、任一SHA/门禁不匹配则停止，保留证据。
不得以本地197项通过跳过AutoDL专用CPU验收或真实前置检查。

资源上限沿用原计划：CUDA allocated8GiB、reserved10GiB、host RSS10GiB；
四个quarter完整恢复checkpoint保留，空间要求现场按B实际大小重算，
不删除历史checkpoint腾空间。非finite、OOM、期限不足或合同失败即停。

## 固定决策分支

- train40各组仍失败：本次预算尚不能完成拟合，回到目标/优化/容量诊断；
  不自动加步、解冻或开启下一炉。
- train通过、dev失败：该预算没有解决跨场景映射，不能称视觉泛化修复。
- train/dev均通过且首动作不回退、B历史复现/新C独立reload均通过：仅离线
  候选成立；后续artifact取回与Gate3/4另按授权处理，M2仍须闭环验收。

此文件不授权启动，未执行新GPU计算。当前SSH授权仅覆盖先前预测取回。
