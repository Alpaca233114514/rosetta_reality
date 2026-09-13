# Hestia 同checkpoint图像K/V场景依赖：执行登记

登记 `hestia-scene-kv-ablation-20260913-001`，承接已提交的同日期scene-kv-ablation-design。用户提供原SSH用于继续诊断，并要求每模块完成单独提交。本轮仅1800次policy forward、0optimizer，无训练/下载/hidden/新Gate或checkpoint选择。唯一轴、十条件、四噪声、train40/dev5、首动作/完整50槽、均值校准与判读规则均沿用设计。

本地实现入口：`hestia_scene_kv.py`、`diagnose_hestia_scene_kv.py`、`run_hestia_scene_kv.py`；stream/verify负责窄回传，`analyze_hestia_scene_kv.py`独立检查保存数组并计算两个checkpoint内的2×2条件效应、MAE/MSE及场景分解。旧模块源码不改动，新workspace identity下执行。

固定条件顺序：640native、1280native、640self、1280self，全部720次控制逐值复现后，按640/1280顺序执行K均值、V均值、KV均值三种干预。每条件180次、batch1，预算300秒；总worker上限2400秒，受保护结束上限3000秒。CUDA分配上限10GiB、RSS8GiB；无新optimizer。实际预算以原生控制计时验证，不扩容或自动重试。

源码现场核验：固定upstream `smolvlm_with_expert.py` SHA `996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf`，`forward_cross_attn_layer`在完整前缀上调用expert的K/V投影后再reshape和attention。K输入已包含VLM侧位置编码，expert查询单独应用RoPE；本轮保留该顺序，仅更换投影输出的真实图像64位置，不将其误称为原始视觉token或固定attention。

native保存按训练40/开发5隔离的两份真实图像K/V校准文件，float32精确承载实际BF16值，分别SHA封存。每层每token每channel以训练40行float64求均值，训练评估另排除自身取39行。开发缓存仅用于原值拷回和身份验证，不进入均值函数。self-copy对照必须原生输出精确一致。记录每个row/layer/KV的完整原生输入/输出摘要、40次重复调用和实际应用位置摘要；finish拒绝任何额外或缺失的干预集合。

执行前固定template中的profile、upstream、历史checkpoint/processor、数据manifest、Action Contract、normalization、全部复用源文件及新增模块SHA。template `launchable=true`仅在本地回归通过后写入；collector和supervisor同时验证科学合同。先在新工作区doctor、workspace-local normalization绑定、CPU检查，任何失败均在权重加载前停下。新的校准统计在native阶段获得，再经原生result/hash绑定供后续条件读取。

回传上限256MiB（归档257MiB）；比此前只保存动作的64MiB上限增加是因为本轮需保存两checkpoint的K/V校准，不改变计算或安全预算。接收器仅允许登记条件的JSON/NPZ和对应日志，拒绝路径越界、链接、重复成员、文件集合/大小/SHA不符。先准备receiver，再启动守护进程；退出落盘、回传匹配收据后走已有受保护关机。不释放实例，不删除旧失败与checkpoint。若回传阻断，保留远端durable证据并按已登记截止结束，不把SSH断开当平台计费证明。

当前只读资源核验：原CUDA卡、无CUDA计算进程，持久盘约5.4GiB可用；源码工作区composite identity `46803787f6a45089e0fc9a73c25a5a62ee19d75c05fea7d829e795471b76b5f6`匹配。SSH连接由Bash启动本机既有客户端；WSL内部客户端的No route to host作为连接失败保留，不当作实例状态结论。环境仍由`configs/runtime/autodl_rtx4090.yaml`及`scripts/run_autodl.sh`记录，`nested_docker_used=false`。

本地固定离线Docker完成161项相关合成/门禁回归、Ruff与格式检查；测试不代表CUDA控制已通过。科学结果在执行前均为not measured。独立分析器只读回传数组，严格区分消除有害场景响应与学会视觉映射；即使离线改善也不改变M2/Gate状态。
