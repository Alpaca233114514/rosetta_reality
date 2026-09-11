# 同一 K/V 的完整 chunk 读出与原生策略输出

登记 `kv-full-chunk-readout-20260911-001`。承接位置正对照通过和首步/末步
目标诊断，检验同一冻结视觉 K/V 是否能用独立读出比已保存动作专家更好地
预测完整 chunk。使用同一 Zen-uniform artifact，不冒充 coverage40 B 重测。

## 固定身份与评价空间

复用 `contextual-kv-depth-20260911-001/features.npz` 中 early 2560 维特征和
原生 `standard_outputs`（45×1×50×14）。原输出来自相同 frame-0 图像/state、
固定零推理噪声，经过原保存的 postprocessor、在安全 clipping 前；不重跑模型。
固定原 40 train / 5 dev，hidden 封存。

从 checksum-pinned 缓存一次查询每场景 frame 0..49，共 2250 个不同数值行，
不调用重复物化首帧的 helper。仅标签/state/timestamp/episode/frame 列，Arrow
先过滤 episode/frame。所有 timestamp=frame/50，原始首动作等于保存标签，
首帧 state 跨场景相同，维度名称/顺序等于原 Action Contract。
未来 state 只检查有限值，绝不作为读出输入。

新比较使用真实训练的 `ActionContractProjectionProcessorStep` 生成标准空间
投影目标，并独立校验 bounds 运算完全一致，超出 source tolerance 即停止。
记录原始到投影的每维改动计数/最大差；原始标签保留。原始标签读出是历史
辅助目标，不等于修复后策略实际学习的投影目标，尤其夹爪不能混用。
策略和读出预测都不增加事后 clipping，不能借裁剪改善评价。

## 读出程序与固定报告

一张首帧的 K/V 对应完整 50×14 标准目标，按 time-major 顺序展平为 700 维。
使用原 train-only 标准化和 dual ridge，固定 expanded alpha 网格
`[0.001,0.1,1,10,100,1000,10000,100000,1000000,100000000]`。
只用 train 五折、seed=20260911、原混合单位 MAE 代理选**一个** alpha，
整个 chunk 共用，不逐时间步调参。关节 radian 与夹爪 normalized 分组报告。
另做五外折/五内折，内层 seed=20260911+外折编号+1；常数逐外折构造。

主要对象是完整 chunk；预先固定辅助窗口 `[0,1)`、`[0,10)`、`[10,25)`、
`[25,50)`、`[49,50)` 和全部逐步误差曲线。报告全部窗口，不根据 dev 选最优步。
每个主组同时满足以下条件，才支持“存在完整 chunk 的读出差距”：

1. dev 读出 MAE 低于同组 native、train mean 和 median，epsilon=1e-8；
2. nested 读出 MAE 低于对应 fold-train mean 和 median，epsilon=1e-8；
3. dev 正确图 MAE 低于全部非自身图各自 MAE 的平均，epsilon=1e-8。

两主组缺一即整体不通过。这个新问题不替代上次位置到左臂末步 `<0.8` 的失败。
原策略在整个训练轨迹上学习，ridge 仅用40个首帧；不能把两者当作等预算训练
对照。只把结果用于定位可读信息与当前原生输出的差距，不证明专家实现有 bug。
dev5 已经参与开发，nested 只隔离 ridge，不是上游模型的新独立测试。

## 运行与停止

现有 digest-pinned Linux Docker / WSL Bash，CPU 2 核，2 GiB 内存和 memory+swap，
网络关闭，分析上限180秒。check_env、Ruff、合成检查、真实缓存前检后仅执行一次。
新读取器真实前检不加载2250行；独立既有缓存测试物化45个首帧，记为预检，
分析自身2250行，无重复。冻结所有调用的项目源码/输入 SHA。
身份、形状、时间、容差、finite 或预算失败即停止，create-only 保存失败日志。
保存 raw/projected 标签、native/ridge 全数组、fold/alpha、指标和 exact array reload。
不读取模型权重、不解码新图、不启动 SSH、optimizer、正式训练或仿真；M2未完成。
