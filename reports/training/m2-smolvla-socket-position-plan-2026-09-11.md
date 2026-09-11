# 冻结 K/V 的插槽像素位置正对照

登记 `socket-position-readout-20260911-001`，2026-09-11。
在已完成动作读出负结果之后，检验同一早层 K/V 是否保留跨场景可读的物体位置。
这是辅助目标正对照，不是改进策略或与动作误差直接比较的学习实验。

## 固定输入与标签

保持既有 Zen-uniform step316、早层 1、2×2、2560 维特征，复用已校验 NPZ。
本轮不加载模型。只从同一 checksum-pinned 本地缓存读取原 40 train / 5 dev
的 frame 0，逐图 SHA 和 episode/state/action 必须与原 K/V 采集记录一致。
hidden 在读表前排除；既有 Arrow 过滤检查及真实缓存测试必须先通过。

像素目标为蓝色插槽在原始 640×480 RGB 图中的颜色掩码质心，顺序 x/y，单位 pixel。
依据官方 ACT 插槽资产的蓝色材质，固定主规则 `B>=80, B-R>=40, B-G>=40`。
固定稳健性规则为 `B>=100, B-R>=60, B-G>=60`，两质心每轴差不超过 2 pixel。
每个掩码像素数必须在 [40,10000]，bbox 宽高在 [4,180]；失败不修改阈值继续。
生成全部 45 图带 bbox/质心的本地检查图，逐图确认落在插槽后才解释数值结果。
像素位置是可观察代理，不是标定的三维位姿，也不证明 peg 的位置可读。

官方参考：
[ACT insertion asset](https://github.com/tonyzhaozh/act/blob/main/assets/bimanual_viperx_insertion.xml)。
此链接只支持颜色选择；本地实际图像和 checksum 才是本次数据证据。
[ACT data reader](https://github.com/tonyzhaozh/act/blob/main/utils.py) 区分 simulation
与 real 数据时间索引；公开 scripted recorder 并不证明本 human 数据的具体采集历史，
不能据此宣布本地存在时间错位。

## 读出与验收

固定 ridge 网格 `[0.001,0.1,1,10,100,1000,10000,100000,1000000,100000000]`，
train-only 五折按两个像素坐标的 MAE 选一个 alpha，seed=20260911。
同样的五外折/五内折作训练集嵌套诊断；外折内 seed=20260911+外折编号+1。
重用上轮已验证的标准化、ridge 和嵌套实现；dev 不选 alpha、不调颜色阈值。

正对照通过要求 x/y 两坐标分别同时满足：
1. dev5 MAE 低于 train mean 和 median 基线各自的 50%；
2. train nested MAE 低于 fold-train mean 和 median 基线各自的 50%；
3. dev 正确图 MAE 低于所有非自身错配图各自 MAE 的平均（epsilon=1e-8）。

若通过，只证明该表示对插槽二维位置可读，排除“这个 K/V 完全不含可读插槽
位置”这一窄假设，不能证明动作专家充分利用该信息。若失败，不证明位置缺失。
报告全部坐标、fold、alpha、逐场景误差与掩码证据，不挑选层或开发场景。
旧模型在 40 train 场景上训练过，nested 只隔离本轮读出；dev5 不是新测试集。

## 运行合同

现有 digest-pinned Linux Docker，经 WSL Bash 启动，网络关闭，CPU 2 核，内存
与 memory+swap 各 2 GiB，分析最多 180 秒。先 check_env、Ruff、合成检查、
真实缓存测试，再做一次采集和读出。输入与代码 SHA 在执行前封存于同名 JSON。
identity drift、提取失败、非 finite、超时或超内存立即停止，保留失败。
新目录保存位置标签、预测、数值记录、45 图检查页；数组 exact reload。
不创建 optimizer，不运行 SSH、模型 forward、仿真或下载。M2 未完成。
