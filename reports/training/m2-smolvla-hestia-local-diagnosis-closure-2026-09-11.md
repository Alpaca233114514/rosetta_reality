# Hestia本地数组诊断收束与下一证据缺口

当前结果：视觉泛化尚未修复，M2未完成。实例由用户确认已关机；本地分析期间
未再次连接、开机、运行远端GPU或训练。每项确认结论独立提交，未推送。

| 已区分的问题 | 实测结论 | 证据 |
|---|---|---|
| 是不是还没完成足够训练 | Hestia确已完成1280步，train各组4/4、dev各组0/4 | `m2-smolvla-hestia-main-recovery-result-2026-09-11.md` |
| 仅仅缺少视觉响应 | C训练响应/对应恢复；开发左关节仍四噪声负相关且误差增加 | `m2-smolvla-hestia-transfer-decomposition-result-2026-09-11.md` |
| socket二维位置超出训练范围 | 4/5开发质心在训练凸包内；近邻动作仍失败 | `m2-smolvla-hestia-geometric-support-result-2026-09-11.md` |
| 图像位置完全没有动作信息 | 两物体低阶坐标对末动作关节有跨场景收益，首动作与夹爪仍不满足 | `m2-smolvla-hestia-object-readout-result-2026-09-11.md` |
| 整段时序偏差解释所有问题 | ±200ms逐场景目标oracle也不能让左关节胜过同样对齐常数；夹爪有可消除的时序分量 | `m2-smolvla-hestia-timing-bound-result-2026-09-11.md` |

以上不会推出“VLM坏了”“必需解冻”“换损失一定有效”或“人工标签不可学”。
共同指向的是拟合后的跨场景左臂映射仍错误；夹爪另有时序问题。将它们合成
一个总loss继续加步数，会掩盖不同失败来源。

目前缺少的是中间保存点的模型级证据，已有数组无法推出320/640/960步的开发
行为。优先取回四个现成checkpoint做本地拟合过程比较，区分随拟合增强恶化
与从早期就未建立映射。具体40文件/约4.48GiB、只读传输与自动关机范围见
`m2-smolvla-hestia-checkpoint-retrieval-plan-2026-09-11.{md,json}`。
尚未获得本次开机/传输授权，当前没有必要为新训练开启GPU。
