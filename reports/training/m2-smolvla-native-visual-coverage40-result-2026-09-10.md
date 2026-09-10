# SmolVLA coverage40 / Hermes 结果：视觉泛化仍未修复

40 场景候选 B 已完成原生 256 步训练。相较旧 8 场景 A，开发集误差下降，正确视觉相对错配的收益改善，但仍未通过预登记视觉验收。**两臂开发验证均为 0/4，M2 未完成，闭环任务成功率未测。** 本轮没有自动启动下一炉。

机器证据见同名 JSON；探索性分解见 `reports/training/visual-coverage40-20260910/posthoc-error-decomposition.json`。四个条件是固定推理噪声，不是四个独立训练重复；五个开发 episode 已参与开发，hidden 保持封存。

## 固定合同与真实执行

- A 复用原 8 场景 step 256 checkpoint，没有重训或续跑。B 从同一 pinned base 新建模型、optimizer、scheduler 和 RNG。
- 只将 frame-0 训练覆盖从 8 扩到既定 40 episodes；batch 4、seed 20260809、256 updates、原生 flow-matching loss、冻结 VLM 和原 processor / Action Contract 不变。
- AdamW peak LR `1e-4`、betas `(0.9,0.95)`、eps `1e-8`、weight decay `1e-10`、clip 10；warmup 16、decay 256、末端 LR `2.5e-6`。256 次实际 LR 和最终 scheduler 已逐项核验。
- 实际消费 1024 个样本，顺序与计划完全相同：A 每场景 128 次；B 为 24 个场景各 26 次、16 个各 25 次。固定计算同时改变覆盖与重复分配，不能拆开解释因果。
- B 训练约 137.17 秒，peak allocated 2,817,605,120 bytes、reserved 2,883,584,000 bytes、RSS 3,708,944,384 bytes，无非 finite 或超资源。
- 冻结审计：345 个 VLM 参数张量数值均不变，145 个 expert 张量中 112 个更新，10 个 projector 张量更新；记录的 BF16→FP32 保存转换逐值无损。完整 recovery state 保留。
- 真实 batch 1/4 forward、样本合同、两步 optimizer smoke 和完整 smoke reload 均通过。A 旧口径指标逐值重现；新口径不覆盖历史 cyclic 错配结果。

## 同一全非自身错配协议的结果

下表为四个固定噪声条件的平均 normalized MSE。C 为正确图，M 为所有非自身错配图各自误差的平均，F 为该集合逐坐标目标均值的解析下限。通过数要求每个条件同时满足 C < M 且 C < F，使用原有 epsilon。

| 集合 / 维度组 | A：C / M / F | A 通过 | B：C / M / F | B 通过 |
|---|---|---:|---|---:|
| train8 / 全维度 | 0.026303 / 0.408706 / 0.192625 | 4/4 | 0.105471 / 0.313326 / 0.192625 | 4/4 |
| train40 / 全维度 | 0.147617 / 0.283589 / 0.186157 | 4/4 | 0.130624 / 0.265276 / 0.186157 | 4/4 |
| train40 / 关节 | 0.153090 / 0.306360 / 0.199330 | 4/4 | 0.133942 / 0.287921 / 0.199330 | 4/4 |
| train40 / 夹爪 | 0.114780 / 0.146958 / 0.107116 | 0/4 | 0.110715 / 0.129407 / 0.107116 | 1/4 |
| dev5 / 全维度 | 0.217349 / 0.198812 / 0.121431 | 0/4 | 0.184360 / 0.187381 / 0.121431 | 0/4 |
| dev5 / 关节 | 0.223464 / 0.197810 / 0.113865 | 0/4 | 0.184314 / 0.186776 / 0.113865 | 0/4 |
| dev5 / 夹爪 | 0.180659 / 0.204823 / 0.166823 | 0/4 | 0.184640 / 0.191013 / 0.166823 | 0/4 |

A 只训练过 train8；其 train40 包含 32 个未训练场景，不能把该行整体称为 A 训练拟合。B 在 train8 的夹爪通过数为 2/4。

两臂全维度训练拟合检查均为 4/4；B 的夹爪单组训练检查仅 1/4，不能声称各组都已拟合。B 的开发全维度 C 较 A 下降约 15.18%，但仍高于 F 约 51.82%，也差于只用 train40 目标均值构造的开发集常数基线 `0.139549`。F 使用开发标签，仅作解析 oracle 下限，不是可部署策略。

跨臂“误差下降且视觉收益改善”检查为 4/4，但这不足以通过整体验收。开发集标准动作空间结果如下：

| 指标 | A | B | 预登记不回退 |
|---|---:|---:|---|
| 完整 chunk 关节 MSE | 0.01063334 | 0.00880458 | 通过 |
| 完整 chunk 夹爪 MSE | 0.01939070 | 0.01874296 | 通过 |
| 首动作关节 MSE | 0.00105296 | 0.00136139 | 失败 |
| 首动作夹爪 MSE | 0.00042063 | 0.00047541 | 失败 |

完整 chunk 夹爪 MAE 也从 `0.05247236` 增至 `0.05654338`；不可只报告 MSE 的改善。A/B 均在独立进程 reload 后对七组完整数组逐值一致，包含完整 50 步预测、目标、噪声和有效 mask。没有通过阈值调节、选择最佳噪声或遗漏夹爪获取正结论。

## 工程修复与负结果分开记录

最初入口的 JSON 科学计数法被原生 YAML reader 读为字符串，改用 YAML 并检查实际 loader roundtrip。随后真实输入暴露历史状态轴 `(B,1,14)`，修正了新检查器的错误 shape 假设，没有 reshape 模型输入。

本次 job 003 在 B checkpoint 审计通过后，原封保存了错误 `AssertionError: Actual 1024 sample order differs`。调查发现 sampler 观察器记录了 1028 个 yielded 索引；实际训练器只消费 256 批。已安装 Accelerate 在 yield 当前 batch 前预取下一批，前 1024 个及后续 4 个索引全部符合计划。

Hermes 单独登记仅评估续接；用真实 `DataLoaderShard` 验证预取行为，完整核验实际训练轨迹后复用 B，不再训练。原失败 closure 未改写。最终 compare 退出码 1 对应已保存的验收负结果；不是模型执行或 reload 失败。

回归在 AutoDL 的独立源码工作区完成：**95 passed，Ruff 全部通过**。新增预取测试包含跨 epoch 的真实 Accelerate 行为、错序、少取、多取及预取尾部错序。执行版本 `02cbf63` 的后续静态检查发现格式和未使用 import 问题；在 `4fbccc6` 整理并验证，运行中的封存版本没有修改。

## 探索性分解与下一步

仅使用已保存的预测，以 `C = 目标场景方差 + 预测场景方差 - 2×协方差 + 均值偏差平方` 分解误差，数值恒等校验通过，没有重新加载模型或数据。

B 开发集关节场景相关系数约 `0.01955`，预测方差 `0.02228`、目标方差 `0.11387`、均值偏差平方 `0.05013`；其视觉相关收益不足以抵消方差和偏差。B 的 train40 夹爪预测场景方差为 `0.00401`，目标为 `0.10712`，均值偏差平方为 `0.01781`。这些结果支持继续调查拟合强度和跨场景映射，不能证明增加步数或解冻必然有效。

下一模块草案为 Athena：先核验只延长训练步数、保持原 scheduler 定义时的尾段行为，设计 40 场景拟合强度诊断。见 `reports/training/m2-smolvla-athena-fit-strength-draft-2026-09-10.md`。当前固定预算覆盖试验按负结果收线；没有自动启动该草案，也没有叠加 paired loss、dropout、unfreeze 或 first-action weighting。

官方材料支持同时重视变化覆盖与重复示范，并按任务调训练步数；它没有证明本项目 40 个 frame-0 样本或增加重复一定有效。[官方指南](https://huggingface.co/docs/lerobot/smolvla#collect-a-dataset)、[固定 v1 论文](https://arxiv.org/html/2506.01844v1)。本报告的具体诊断来自上述项目证据。

## 身份与收线

- B 训练源码：`e11c7aaddc8c1eab84365d399ef8696cb5d0f83b`。
- A/B 续接评估源码：`02cbf63269c2f3038e9e2cb7ec1af64d84b17813`。
- 回归/格式整理源码：`4fbccc6bacea682759a29e12b28da8b0ab24ef9b`。
- A model SHA-256：`fd95a19ca8d8f9e1c6a5c477807aaf69fc3e19bb15e626bd53e4224da8ad3a5c`。
- B model SHA-256：`4e91624b34c3a16e22e0cbba19d8bcf3352a0d3835522ea6973ee22f0397accf`。
- B 完整 checkpoint：`checkpoints/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/smoke/m2-smolvla450m-visual-coverage40-main256-003/checkpoints/000256`。
- 评估证据工作区：`dev/visual-hermes-post-20260910-02cbf63/runs/visual-hermes-coverage40-eval-001/`；训练原件保留在 `dev/visual-coverage40-old4090d-20260910-e11c7aa/`。

2026-09-10 15:54:31 CST，核验 worker 已退出、GPU 无计算进程、reload 和证据均保存后，调用 SHA 固定的官方关机 wrapper，SSH 随即由远端关闭。未释放实例，平台计费状态未独立核验。完整 checkpoint 和预测留在 durable storage，本地仅回收小型指标与哈希；未执行本地模型 reload、新 Gate 3/4 或公开权重发布。
