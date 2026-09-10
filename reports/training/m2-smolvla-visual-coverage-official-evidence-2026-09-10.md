# SmolVLA 场景覆盖试验：官方依据核对 — 2026-09-10

结论：8→40 个 train episodes 的 frame-0 试验有调查价值，但有效性是项目假设。
官方材料没有验证本项目的 ALOHA insertion、256 步、固定样本或视觉对照验收。
本次仅查阅公开技术文字和本地文档，没有访问服务器、模型或数据。

## 官方材料实际支持什么

| 来源与定位 | 官方内容的简述 | 对本项目的意义与限制 |
|---|---|---|
| [LeRobot SmolVLA 指南：Collect a dataset](https://huggingface.co/docs/lerobot/smolvla#collect-a-dataset) | 建议约 50 条示范作为起点；其抓放例子覆盖 5 个位置、每位置重复 10 次，25 条版本表现不佳。 | 同时重视变化覆盖和每种变化的重复。40 个 episode 未必等于 40 种独立视觉场景；frame 0 也不是完整示范。 |
| [同指南：Finetune SmolVLA on your data](https://huggingface.co/docs/lerobot/smolvla#finetune-smolvla-on-your-data) | 给出 batch 64、20k 步的示例，说明步数应按任务效果调整。 | 这是示例，不能替换已冻结的 batch 4 / 256 步，也不是本项目的算力授权。 |
| [SmolVLA v1 论文 §3.1](https://arxiv.org/html/2506.01844v1#S3.SS1) | 图像、语言和状态经 VLM 形成条件特征；action expert 用 flow matching 预测动作块，交替使用 cross/self attention。 | 原生目标包含噪声动作条件；teacher-forced loss 下降不能单独证明推理时正确使用图像。 |
| [同论文 §4.3](https://arxiv.org/html/2506.01844v1#S4.SS3) | 实现采用 frozen VLM、训练 action expert、50 步动作块和 10 步推理去噪；仿真每执行一步后重新观测。 | 支持先保留原生架构/损失/冻结边界；不能推出当前 frozen-VLM 一定足够，也不能把论文描述代替项目参数更新证明。 |
| [同论文 §4.1、§5.1](https://arxiv.org/html/2506.01844v1) | 以任务成功率评估；讨论数据多样性、规模、VLM 适用性和长任务的限制。 | frame-0 离线对照只能支持受限的开发进展，不能完成 M2。 |

查阅日期为 2026-09-10（Asia/Shanghai）。指南是可变页面；论文固定为 v1。
项目执行继续服从自身 revision、源码和 processor 的 hash-bound 身份，
不因在线指南更新而升级依赖、改变去噪时间方向或复制新的 CLI 配方。

## 本项目的假设与设计选择

1. **工作假设**：训练集内已能学到图像与动作的对应，但跨场景失败可能部分来自
   frame-0 覆盖太少。这来自用户交接的 8 场景结果，不是论文对本项目的诊断。
2. **固定预算干预**：从 8 扩到既定 40 个 train episodes；两臂均为 batch 4、
   256 次更新，即 1,024 次主试验样本暴露。均匀采样下每场景平均暴露从
   128 降到 25.6。它改变覆盖与重复分配，不能拆开两者的因果效应。
   固定更新/暴露数也不等于实测墙钟或解码成本完全相等。
3. **保守对照**：保留旧 8 场景 checkpoint，仅重新评估；40 场景臂从相同
   pinned base 全新初始化。没有第二个训练 seed，也不是独立复现实验。
4. **新评估口径**：两臂都用所有非自身图像错配的平均误差；旧 cyclic 数字
   仍单独保留。平均的是每个错配预测的误差，不是先平均预测再算误差。
5. **均值下限**：仅在非视觉条件完全相同、目标坐标/有效 mask 相同的集合内，
   平方误差的最佳固定动作块是该集合逐坐标目标均值。开发集均值是使用开发
   标签构造的解析 oracle 下限，不是可部署基线、train-only normalization
   或独立测试成绩。MAE 的最优常数通常是中位数，不能套用均值下限。
6. **结论范围**：五个 validation episodes 已参与开发。四种噪声条件不等于
   四个独立训练重复；所有错配组合也不增加独立场景样本数。正结果仍须另行
   登记完整轨迹和闭环验收；负结果只关闭该固定预算设计，不否定所有覆盖假设。

## 审计材料如何约束这一步

- Faust 审计的 T3/T4 要求固定推理噪声、区分图像敏感性与有效视觉利用；
  T5 提醒样本暴露、更新次数和优化路径不能混为一谈；T6 要单看夹爪的
  内部支持域，不能仅看有界解码；T7 要求保留完整恢复状态但不能冒称已证明 resume。
- Zen 已否定其登记尺度的 first-action weighting 足以改善闭环这一主张；
  本轮保持原生损失，不把它重新叠加进去。Zen 审计的“唯一失败项”文字与
  uniform 的 5 次 joint-limit violations 有冲突，架构入口已说明以 Gate JSON 为准。
- vfunfreeze 的帧 0 对齐通过后仍 Gate 4 `0/5`；其历史“表示轴耗尽”措辞只
  能覆盖已测配置/规模，不能解释成所有视觉学习假设已被穷尽。
- 本地 20 项 paired-loss 算术测试和此前视觉缓存诊断均是不同模块证据。
  此次计划不启用 paired loss、dropout、unfreeze 或新的特征缓存。

具体可审阅方案：
`reports/training/m2-smolvla-native-visual-coverage40-plan-2026-09-10.md`
及同名 JSON。远端旧报告未在本轮读取；其状态和数值仅按用户交接标注，
不以本地 HEAD、历史关机记录或本地回归结果替代远端事实。

后续 SSH 核验已找回原始报告和控制 checkpoint，详见
`reports/training/m2-smolvla-native-visual-authority-2026-09-10.md`。
以上“未读取”描述保留为资料查阅阶段的范围，当前证据状态由后续核验报告更新。
