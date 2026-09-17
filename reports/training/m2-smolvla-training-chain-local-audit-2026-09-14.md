# SmolVLA 本地训练链审计交接 · 2026-09-14

状态：按用户低电量要求停止扩展并收尾。**全面审计尚未完成，未证明闭环问题已修复。** 全程本地，无 AutoDL、下载、正式训练、提交或推送。

## 已落地的修复

- v2 启动器与训练上下文新增实际实现文件 SHA 核对，且必须密封核心启动文件。静态读取历史计划仍可进行，旧 hash 不被改写；未来运行必须生成新身份。
- 扩展清理尽力执行全部卸载，保留卸载失败项和原始训练/安装异常；Trackio 清理不再被前一卸载异常跳过。
- 拒绝未实现的 resume 和非 1 梯度累积声明，避免计划写了但实际静默按另一种行为运行。
- 验证集限制在父配置注册集合内；拒绝重复验证帧；checkpoint 声明网格必须包含最终步。
- masked-camera skip 每次 forward 都检查 mask，使用张量断言保留编译路径检查。
- 梯度诊断按 action expert、VLM、state/action projection 分组，避免全部折叠为 model。
- observer 分别报告完成暴露数和唯一输入帧数。
- 新增独立 `explicit_sample_schedule` feature 与覆盖工具，保留首帧历史路径；跨时段顺序有 SHA、split、seed、预算及轨迹边界约束。

## 已测证据

- 修复前原有基线：103 passed。
- 第一批 9 个反例在原实现全部失败；修复后相关 73 项通过。
- 相机 mask 与梯度归组两个反例在原实现失败；源码 SHA 缺口另有启动器反例。
- 动作/loss/状态扩展/Iris 回归：104 passed。
- 身份、mask、梯度、小模型原生更新与恢复链相关检查：53 passed。batch 1/4 的 tiny CPU policy 连续/恢复样本、loss、梯度、LR、AdamW 状态和参数一致；**不等于正式 SmolVLA resume 通过**。
- 扩大回归：480 passed、9 failed、2 deselected。1 个失败是新增 feature 注册表预期遗漏，已补，未重跑；其余 8 个停在 Zen/vcdropout/vfunfreeze 历史源码 pin 校验，保留，不更新历史 hash、不关闭检查。
- 最终 18 个变更 Python 文件通过 AST 解析及 Ruff。最后只做了导入排序/格式化，没有重跑 ML 测试。
- 源码清单覆盖 285 个直接及递归依赖 Python 文件；这是 AST/导入清单，**不是 285 个文件逐一语义审计全部通过**。

## 数据核验

- 固定本地缓存：训练 20,000 帧、开发 2,500 帧；逐 episode、Arrow 过滤后流式检查。隐藏样本物化数 0。
- 索引连续唯一、50 Hz 时间对齐、state/action shape/finite、动作越界容忍与变换检查通过。
- 非隐藏集共有 6,973 个动作元素需要合同限幅，均在既定源数据容忍范围内；这不是数据“完全没有异常”的证明。
- 405 个跨时段图像样本解码、原始 state/action 对照、动作块首项、尾部 padding 与指令核验通过。其余图像帧未逐帧解码。
- 首帧控制：每臂 5,120 次暴露，40 个唯一输入帧，2,000 个唯一动作目标帧。不能称训练遍历 25,000 帧。
- 等预算跨时段草稿：每轨迹 128 个等距帧含首尾，共 5,120 个唯一输入；保持控制 episode 顺序。覆盖全时段不等于遍历全部输入帧。
- 保存 processor 的全训练归一化复算在 Arrow 空批次处失败。已加空批次跳过，**未重跑**；不得声称 saved processor 统计已复核通过。

## 本地证据入口（仓库相对路径）

- `runs/training-chain-audit-20260914-001/`：数值审计、405 图像记录、两臂采样顺序和评估样本。
- `runs/training-chain-inventory-20260914-001/inventory.json`：检查时源码 SHA、导入关系及已安装上游源文件快照。
- `runs/training-chain-processors-20260914-001/result.json`：归一化复算失败证据。
- `runs/training-chain-finalize-20260914/static-result.json`：最终静态检查。

## 下次继续，勿直接训练

1. 重跑最终改动的聚焦回归；为历史测试绑定原始源码 fixture/隔离 checkout，保留真实 pin 检查，区分原有失败与本轮新增影响。
2. 验证 processor 复算脚本的空批次修正和保存张量键名，再检查保存统计与完整 processor 的实际输出。
3. 新采样 feature 仅完成基础 sampler 测试；补齐真实 feature 安装、observer 与 batch/padding 集成。两臂可启动训练配置、预算与完整生命周期**未生成完成**。
4. 整理剩余入口的逐项语义审计。已发现但尚未修复的历史选择器问题：Zen/vcdropout/vfunfreeze 声称平局选较早 checkpoint，代码却按负 step 选较晚；历史 exporter 的七项指标相等也不能推出全张量相等。保留历史文件，以新版本入口修复，不回写历史结论。
5. 继续完整模型本地 forward/reload 与正式恢复能力核验；CUDA、全量图像解码和 Gate 重测均未执行。

最新已有 Iris Gate 证据仍为：两臂 Gate 3 通过，Gate 4 各 0/5；M2 未完成。
