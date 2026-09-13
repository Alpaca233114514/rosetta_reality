# Hestia 原生视觉前缀路径：等待 SSH 的采集计划

状态：本地实现与合成检查阶段。用户要求追到根因，允许自主规划本地工作，
需要 SSH 时联系；本文件不表示已经获得外部 GPU 开机/执行/关机授权。
名称 `hestia-prefix-path-20260912-001` 已检索无重名；不是新训练炉。

## 为什么需要新的原生证据

阶段桥接已表明：单独对齐左开合时机不足以消除原生 C 的朝向失配；
给几何读出真实时刻、还原同一原始时间目标，也未共同通过完整动作。
需要区分原生冻结表示的可读信息与动作专家投影之后保留的信息。
既有 Zen 特征不能冒充 C 的实际同环境激活，已有 XPU BF16/FP32 均未通过
原 CUDA 容差，因此不能用新的 XPU 观测替代原生 CUDA 因果链。

## 冻结采集

仅原 Hestia final1280，原 40 train/5 dev frame0、相同完整 50 槽、四个保存
噪声、180 forwards。原模型/数据/processor/normalization/Action Contract/
Torch2.8.0+cu128/4090D 身份全部沿用已成功 CUDA 曲线；无权重修改或 optimizer。

在第 1、15 个零基 cross-attention 层的 k_proj/v_proj 上增加只读前后 hooks。
保留实际 leading 8×8 real-camera tokens，分别记录原生 VLM K/V 输入和
专家投影输出，保留完整 token，不先 pooling；每行八个数组、45 个 NPZ。
拒绝 token grid、attention layout、width、mask、shape、finite 漂移。
每个 tensor 在全部 10 denoise steps 和四噪声之间须逐值相同。
prefix 原始数组预算 48 MiB，整包含旧 collector 结果仍限制 64 MiB。

复用原 collector 的所有身份、当前非视觉恒定、原始图像 hash、模型参数
前后不变检查。首个 normalized 控制必须精确复现，完整180个 normalized
及 standard 输出也须与历史逐值相同（atol=rtol=0）；hooks 返回 None，
不替换张量。失败停止并保留证据，不自动重试或换后端。

## 守护与回传

`scripts/run_hestia_prefix_path.py` 是旧守护器的独立派生，仅将允许 step
收紧为 `[1280]` 并替换采集器及其测试入口。保留显式 inference/shutdown
permit、独立 watchdog PID+start_ticks、孤儿子进程清理、无其他任务检查、
当前 AutoDL doctor、workspace-local normalization 绑定与真实预检。
总工作期限480秒，最迟600秒走原 SHA 绑定的受保护关机入口；不释放实例。
受控输出只写新 content-addressed workspace 的新 job，不覆盖旧 authority。

`scripts/receive_hestia_prefix_results.sh` 为 WSL Bash 接收入口，host/port/
workspace 显式传参。流只含严格白名单的 prefix/原生结果；本地离线固定
Linux Docker 验证 path/type/size/set/SHA 后回传 matching receipt。
训练/下载/删除/提交/推送均不在此范围。连接信息和凭证不写计划或源码。
当前不启动 SSH、实例或接收器。

## 回传后的固定检验与归因限制

按真实 metadata 重排同一45场景，pre/post、early/late四处分别采用固定
mean pooling 与2×2 ordered pooling，训练嵌套选择 ridge alpha（沿用七值
网格及seed20260912），目标先双物体位置、再左右开合时刻/原生动作，
另做真实阶段的位姿控制。初始输入和目标不变；dev 只作诊断，不选层或
pooling，不把失败探针解释为信息不存在。完整token为后续受控空间诊断保留。
若位置可读、动作失配，定位已拟合映射；若投影前可读、后不可读，仅将
投影列为待干预位置，还须实际干预验证，不能据相关性宣布根因。
新的训练、策略部署、hidden 或 Gate 3/4 均未授权。本次采集本身不保证
确定唯一根因；整个根因目标保持进行中。
