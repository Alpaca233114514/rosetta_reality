# 实际coverage40 B：共同偏差与全局时移的两个独立对照

先决条件已通过：历史A/B七数组取回，完整原口径比较JSON的SHA逐字节重现。
本轮直接使用B step256的既有standard预测/目标，不重新加载模型、图片或数据集。
范围不扩到新训练、GPU、hidden或Gate。

登记两个独立运行：`coverage40-bias-20260911-001` 与
`coverage40-time-shift-20260911-001`，每次仅应用一个处理，不组合两者。
使用保存的45个frame-0场景、50槽位、14维；前40train，后5dev。
目标沿用Hermes原生processor后再解码的standard_targets，保留其浮点roundtrip，
不替换为Zen数组或原始未投影标签。先核验合同与原生控制合法性。

处理定义直接复用已公开的两个诊断：

- bias：各噪声分别按train40残差均值得到每槽位/维度的共同加性偏差。
  校正LOO每行只用其余39行残差；candidate与control按同一Action Contract裁剪。
- time_shift：各噪声分别只用train完整chunk混合MAE，从[-10,10]选单一整数lag，
  所有场景/动作维度共享；同分先绝对值小，再负值。边界保持端点，所有目标
  仍评分。LOO每次只用39行选lag。网格边界不扩展。

zero与三个固定Gaussian seed 20260905/06/07全部分别报告；不挑最佳噪声。
每噪声两物理组必须同时满足：完整dev MAE下降至少20%、低于train mean及
median、正确图MAE收益为正、校正LOO低于两种对应常数、首动作MAE不回退。
阈值容差沿用1e-8。跨噪声条件沿用项目固定四噪声的至少3/4同时通过，
仍只称有限离线诊断通过，不能称视觉泛化或任务修复。

保存full/first/early/middle/late/last窗口指标、每场景误差、共同偏差/lag与
LOO值、clipping计数、处理前后完整数组及exact reload。每个noise的偏差/lag
可不同，这是分别检验固定噪声条件；没有验证可部署的通用校正器。
两组单位分别为radian和normalized。LOO只隔离新校正，不消除上游B训练暴露。

输入/源码SHA在对应JSON冻结，复用已验证的read_bundle及校正函数。
先Ruff、格式和相关合成隔离检查。每个运行独立上限180秒，现有Linux Docker
离线CPU2核、主存/memory+swap各2GiB；输出create-only，失败不改门槛。
结果不论正负都单独提交，不改历史A/B验收0/4；M2未完成。
