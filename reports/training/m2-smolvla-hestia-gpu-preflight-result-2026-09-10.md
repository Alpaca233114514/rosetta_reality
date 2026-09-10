# Hestia GPU 预检结果

状态：**工程预检通过，视觉泛化仍未修复，M2 未完成。** 本次只执行独立两步 smoke，没有启动 1280-step 候选，也没有新的 B/C 开发视觉评分或 Gate 3/4。

源码 `31f3c0b1fea272d8f82d93d4e6b0533bbe09d672`；独立工作区 `dev/hestia-gpu-preflight-20260910-31f3c0b`；输出 `runs/hestia-gpu-preflight-002/`。原始小证据已按 SHA256 回收并归档到 `reports/training/hestia-preflight-20260910/002-*`，报告 JSON 给出映射。模型、数据和 checkpoint 没有经过本机传输。

## 已验证

| 项目 | 实测结果 |
|---|---|
| CPU 代码回归 | 101 passed，6.30 秒，无 skipped；Ruff 通过 |
| 真实数据检查 | 1 passed，6.40 秒；doctor、benchmark 通过 |
| GPU | RTX 4090 D，24,564 MiB；本次 GPU UUID 已登记；无嵌套 Docker |
| 原生 no-optimizer forward | batch 1、batch 4 均通过 |
| 真实输入合同 | 45 个非 hidden frame-0 图像不同；非视觉条件相同；动作语义与训练视图索引通过 |
| 实际训练观察 | 交付 8 样本、完成 8 样本、成功 optimizer step 2 次；没有未完成 batch 或观察错误 |
| 参数更新 | VLM 345/345 未变；expert 112/145 改变；projectors 10/10 改变 |
| checkpoint | step 2、processor normalization 与恢复状态检查通过 |
| 独立 reload | 不同 PID 33292 / 44860；每进程 45 场景 × 4 噪声 = 180 组；normalized 与 standard 两组完整 50-step chunk 数组 dtype 和数值逐项完全一致 |

观察到的两步 LR 分别是 `0.0001`、`0.00005125`。这是原生 launcher 的独立短 smoke 调度，不能当作主候选 1280-step / warmup-16 调度的前两步证据。主候选仍须记录并核验完整实际 LR 序列。

VLM 的 BF16→FP32 序列化转换经逐 tensor 核验保持值完全一致，不计为学习更新。两步权重 SHA256：`deabafebcc11cfca939d6b3940bf1421b90bebfa4648709cd15a854d954b987a`。

checkpoint 位于 durable root 下：
`checkpoints/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/smoke/m2-smolvla450m-visual-hestia-smoke2-002/checkpoints/000002`。

GPU worker 从启动到独立 reload 完成约 **289.92 秒（4 分 50 秒）**，低于注册 1200 秒计算期限。各受监测阶段峰值：allocated 2,812,718,592 bytes（2.62 GiB），reserved 2,910,846,976 bytes（2.71 GiB），host RSS 3,964,416,000 bytes（3.69 GiB），均低于 8/10/10 GiB 上限。完整资源分阶段数值见同名 JSON。

## 失败保留与关闭状态

001 在 Ruff 阶段停止，GPU 模型未运行；002 首次 CPU 门禁又发现新增 hash 行超长。两份源码和旧工作区保留，修正后才得到上述 101 项回归与真实 GPU 结果。不能把较早的格式化成功或历史测试通过当作当前代码验收。

supervisor 记录 worker 退出码 0，并按登记保留 120 秒回收窗口后执行平台关机流程。回收窗口之后的一次 SSH 检查返回连接关闭。关机请求文件未能回收，平台电源/计费状态未独立核验；没有请求释放实例。

## 下一步

当前完成的是评分器与原生更新观察器的工程验收。`visual_fit.py` 的七数组 B/C 协议已通过合成回归，尚未用真实 B/C 模型证据运行；本次 smoke 的两组预测数组 reload 不能替代该项验收。

接下来补齐真实 B/C collector 和训练完整性封存入口：绑定实际 saved config、processor 文件清单、固定 split 与输入；复用 B 的原 checkpoint，不重训；候选 `m2-smolvla450m-visual-hestia-fit40-001` 从固定 base 新建 optimizer，1280 updates / 1280 decay、40 个 frame-0 场景。保持同一组噪声下关节/夹爪/aggregate ≥3/4、标准全 chunk/first-action 非回退、七数组独立 reload 的门槛。

执行主候选前重新核验空间与原生源码/环境，按已保留 smoke 和待新增 checkpoint 分项计算磁盘预算，避免把同一份既有 smoke 重复计入新增需求。仍须封存主候选期限、训练消费 5120 样本、全部 LR、冻结/更新审计与输出清单。训练拟合或开发门槛失败即保留负结果并停止，不自动换 loss、dropout 或 unfreeze。
