# Athena 静态核验：原草案不可启动

已核对当前功能分支 `d3fb016419d20fcd560959d41b7fac4c6987b80b`。上轮 coverage40 / Hermes 的开发验收仍为 0/4，本轮没有模型执行、SSH、开机或新训练。原 Athena 草案保留，以下核验关闭其中不能成立的假设。

## 已核验的源码事实

从 Hugging Face 官方仓库读取固定 revision `c903b114a90e703b3f7d0c46cb38727c328c55ff` 的三份小型源码文本，未执行这些文件。其 SHA-256 与此前真实 worker 核验值一致：

| 官方文件 | SHA-256 |
|---|---|
| `optim/schedulers.py` | `05d57770348fbb3f412f52804a8d0f015f8128a37c55d5ce8ed9af6ed8522bc9` |
| `scripts/lerobot_train.py` | `4d15d283ea54583f552b32088db0b6c195250905ca6daf06d4670383790e2059` |
| `datasets/sampler.py` | `f715aaaa1118ef8901928f92975f7bc08bca412860f072bbcca84555303cc38e` |

原生 cosine scheduler 把衰减位置截断在 decay steps，故在其定义不变时，超过 256 后保持 `2.5e-6`；这由 SHA 匹配的官方源码证明，不是本轮重新执行 optimizer 得出的结果。[固定 revision 源码](https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/optim/schedulers.py)。

然而项目 `src/rosetta_reality/vla/training/plan.py` 的 `validate_optimizer_contract()` 要求 decay steps 等于 training steps；原草案的 training 1280 / decay 256 会被该既有合同拒绝。已有 schema 回归覆盖不匹配时拒绝，但本轮没有重新运行测试。不得移除这条检查来启动草案。

`src/rosetta_reality/vla/fixed_visual_samples.py` 只接受 smoke phase 与 `bounded_visual_overfit`，正式 train 入口映射到 formal phase，会被固定 frame-0 样本合同拒绝。不能把 CLI 从 smoke 换成 train 后仍宣称采样协议未变。现有 runtime profile 也不提供默认正式训练许可。

## 修订候选的优化路径

新的候选是 **1280 次更新 + 1280 步 cosine decay**，warmup 仍为 16，其余训练超参数不变。这是“增加训练预算并延展退火”的联合干预；不是固定 scheduler 的纯重复次数对照。它只适合作为拟合强度诊断，不能用于分离训练次数与优化路径的因果贡献。

按原生公式做静态算术：

| optimizer 更新索引，从 0 开始 | B，decay 256 | 候选，decay 1280 |
|---|---:|---:|
| 0 | 0.00000588235 | 0.00000588235 |
| 15 | 0.00009411765 | 0.00009411765 |
| 16 | 0.00009906328 | 0.00009996242 |
| 128 | 0.00005125000 | 0.00009761401 |
| 255 | 0.00000250367 | 0.00009075979 |
| 1279 | 超出 B 训练范围 | 0.00000250015 |

只有前 16 次更新的 LR 相同。原草案“先确认候选 step 256 权重与 B 完全一致”的要求与该修订设计冲突，应当撤销该候选的错误预期，改为分别验证各自注册的完整优化轨迹；不修改 B 的历史一致性证据。机器算术及源码 hash 见同名 JSON。

## 预取尾部不是永远多一批

Hermes 的 1028 yielded / 1024 consumed 只适用于 B 结束于 epoch 中途的特定位置。已保存的实际 Accelerate 源码显示：遇到 epoch 最后一批时，下一次底层读取先返回 StopIteration，再 yield 最后一批，因而没有下一批预取。

40 样本 / batch 4 每 epoch 10 批。候选 1280 steps 恰好结束于第 128 个 epoch；静态推断末尾应是 **5120 consumed / 5120 yielded**，并非 5124。该规模的真实 DataLoaderShard 行为尚未运行；必须用已安装类和原生 sampler 验证，不能把 Hermes 的固定 `+4` 检查照搬。

后续观察器应分别记录三层：sampler 生成序列、`cycle(dataloader)` 实际交付给训练器的原始 batch 身份、成功 optimizer update 的计数与 LR。官方训练器在取得 batch、预处理后才调用 update；失败于预处理或更新的 batch 不得计为成功训练暴露。[固定训练器源码](https://github.com/huggingface/lerobot/blob/c903b114a90e703b3f7d0c46cb38727c328c55ff/src/lerobot/scripts/lerobot_train.py)。只包装被审计的入口作记录，须证明不改 RNG、batch 内容和 native 更新行为。

## 继续工作的实际入口

下一阶段先在无卡 AutoDL 上完成 CPU 合同预检：实际 schema/launcher、原生 scheduler、真实 DataLoaderShard 的 epoch 边界和读写观察器回归，以及 disk/身份核验。通过后才能完成新的 hash-bound 执行合同和单独 GPU 监督器。上轮 95 项回归是上轮结果，本轮不冒称已运行新的测试。

修订方案与逐项未测门禁见 `reports/training/m2-smolvla-athena-fit-strength-revised-draft-2026-09-10.md`。它仍不可直接启动。当前真实阻塞是需要恢复可用的已登记 Linux 执行环境；不以 Windows Python、未登记本地 ML 环境或修改门禁替代。
