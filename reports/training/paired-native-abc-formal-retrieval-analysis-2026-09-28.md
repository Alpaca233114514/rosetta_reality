# Paired native A/B/C 训练回传与 Astra High 分析（2026-09-28）

运行身份：`paired-native-ABC-formal-20260927-001`。本报告只解释昨晚已完成的训练及回传文件；未运行新模型计算、独立重载或策略评价。原始实验、准入、训练配置和失败记录均保留。

## 回传与完整性

- 远端 run 目录回传至 `.work/paired-native-ABC-formal-20260927-001-received/paired-native-ABC-formal-20260927-001/`。远端生成的逐文件 SHA256 清单位于同一接收根的 `remote.sha256`，其 SHA256 为 `d98628f46bdf537647b721157e4d639594e235b9561061a86c07bca3a5538dfa`。原始普通文件 **250/250** 本地逐文件核对通过，共 **19,379,600,018 bytes**；见 `verification.log`。
- 正式训练引用的源 bundle 另回传至同一接收根的 `source-bundle-20260927-003/`，其中普通文件 **64/64** SHA256 核对通过；清单 SHA256 为 `4c0954af1926f29bf907975603985ac8978e3360536ad0598ecb8268a42dc491`。先行回传的元数据副本为原 run 文件的子集，**58/58** 核对通过。
- 远端三臂 `last` 链接均指向 `005000`。Windows 解包形成目录副本，逐文件比对各 **16/16** 与对应 `005000` 相同；它们不计为新 checkpoint 或独立备份。远端原始普通文件与这些副本须区分。
- 12 套 checkpoint（A/B/C 各在 1250、2500、3750、5000 步）均有 16 个原始普通文件，包含模型、processor、tokenizer、optimizer、scheduler、RNG 与指标。保存的训练步、scheduler epoch 和指标步一致；只读 safetensors 文件头检查显示每套 500 个模型张量、465 个 optimizer 条目及 8 个 RNG 条目，声明的数据范围未越过文件末端。

## 训练事实和比较边界

`training-{A,B,C}/completion.json` 与 `worker-exited.json` 记录三臂各 **5000** 次成功更新，所有队列阶段退出码为 0。源 bundle 的逐更新文件各有 5000 个连续序号和实测 batch mask。三个初始化记录的 500 个张量名称、shape、dtype、hash 逐项相同，公共 **15,000** 条 D 采样记录与学习率序列逐项相同，终点学习率为 `2.5e-6`。每臂最终 122 个张量内容 hash 改变，345 个冻结 VLM 张量未变；B/C 的 122 个变化张量终值不同。

| 指标 | A：D | B：D+R | C：D+S |
| --- | ---: | ---: | ---: |
| 成功更新 / 输入曝光 | 5000 / 20000 | 5000 / 20000 | 5000 / 20000 |
| 唯一输入帧 | 16227 | 15203 | 15203 |
| 有效目标 slots / padding slots | 951354 / 48646 | 963303 / 36697 | 963303 / 36697 |
| 最后 500 步日志 loss 均值（约） | 0.0861 | 0.0975 | 0.0972 |
| preclip norm > 10 的更新数 | 11 | 119 | 118 |

计划曝光的独立重算与实际有效/padding 计数吻合。训练耗时分别约 1061.94、1057.86、1070.92 秒，均早于收紧后的总截止。B/C 早期梯度裁剪事件明显更多，但末尾 1000 步三臂均无 preclip norm > 10 的记录。postclip 峰值分别为 10.0131、10.0112、10.0106；不能表述为严格不超过 10，也不能据此断言裁剪失效。

这是 **探索性** 单 parent 比较：`N_strict=0`、`N_exploratory=1`，新增数据来自 parent 2436。203 个辅助起点中，154 个在共同前缀，49 个在扰动后；5000 次辅助曝光中只有 1210 次来自扰动后起点，占整臂 20000 次输入的 **6.05%**。B/C 的有效监督相同，但较 A 多 11949 个有效 slots（约 1.256%），且固定预算内替代了部分 D 曝光。本轮 A 只有 16227 个唯一 D 输入，不能拿历史 canonical 的 20000 个唯一输入及其 Gate 结果充当本轮 A 的评价。

**现有数据无法判断恢复数据是否提升策略。** 三臂 loss 来自不同训练分布，不是共同任务指标；B/C loss 接近也不代表模型相同。唯一 parent、单训练 seed、成功筛选、未建立的 legacy 场景不重合及尚未整体合格的 teacher 都限制泛化解释。

## 尚未完成的测量

固定本次三臂 `005000` 终点，先在独立进程中做完整预测数组重载，再以相同运行环境、reset/噪声与安全停止协议，测 A/B/C 五场景和共同恢复 bank。不要根据训练 loss 后验挑选 checkpoint。现有两步 smoke reload 不能证明这三个 5000 步终点的完整输出一致；没有本次训练后的策略评价、正式 Gate 4 或 M2 完成证据。

恢复 bank 应保留十个预登记场景的完整状态表：目前八个真实分支、两个未产生，仅 2445 具有已验证恢复到终局成功链；2443/2446 的扰动未达验证阈值。`evaluate_paired_policy_v2.py` 明确标记其独立测量不等价于原正式 Gate 4。平台当前计费/关机状态未在本次 SSH 文件回传中经控制面核验，且本轮未发起关机。

主要证据：接收根的 `verification.log`、`source-bundle-20260927-003/verification.log`、原 run 的 `worker-exited.json`、`training-{A,B,C}/completion.json`、三个训练日志/gradient JSONL、源 bundle 的 `schedule-{A,B,C}-train.json` 和 `paired-mixture-{a,b,c}-train-20260927-008-actual-*.json`；数据边界见 `reports/training/paired-real-input-and-exposure-verification-2026-09-27.md`、`reports/training/paired-q2-v3-cached-grid-independent-verification-20260927-001.json` 及 `reports/training/paired-bank-v3-state-independent-verification-20260927-001.json`。
