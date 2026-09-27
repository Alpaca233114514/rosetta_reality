# 未训练／训练后模型对比：本地静态前置检查

本轮选择复用现有 canonical 检查点，暂不重训。静态文件身份检查通过，
逐参数数值比较、真实模型 trace 和新 Gate 尚未执行。Docker 返回
`Docker Desktop is manually paused`，容器阶段等待恢复；本轮未使用 SSH。

## 本轮实际检查

- 工作分支为 `codex/closed-loop-reproducibility-20260918`；保留进入任务前的
  全部修改、TorchLens/Basin 工作和历史证据。没有提交、推送或修改训练代码。
- 重新计算原始 base manifest 中的 10 个文件，以及 canonical 两个恢复点的
  33 个文件 SHA256，**43/43 与既有记录相符**。包含模型、配置、处理器、
  tokenizer、优化器/RNG/scheduler 恢复文件；只读取字节核验，没有反序列化权重。
- 机器证据：`reports/training/m2-smolvla-prepost-preflight-2026-09-23.json`。
  校验脚本：`.work/prepost-preflight-20260923-001.ps1`。输出采用 create-only。
- 原始 manifest 与 checkpoint receipt 是比较依据；本次证明本地文件与这些
  既有清单一致，不额外宣称清单获得了上游签名或远端当前状态认证。
- 已阅读架构、Faust/Zen 审计及 canonical 004/005 完成报告。
  引用对话读取结果仍有截断，不能假定省略部分是完整实施规范。

| 对象 | 本地位置 | 本次核对的模型 SHA256 |
| --- | --- | --- |
| 原始 base | `models/lerobot--smolvla_base/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors` | `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| canonical 2500 | `runs/canonical-checkpoints-received-20260914-002/step-002500/verified/pretrained_model/model.safetensors` | `a3494600183d0a24c71ab7db0976afbc57567ac714ac678fd572c607e06a2f0c` |
| canonical 5000 | `runs/canonical-checkpoints-received-20260914-002/step-005000/verified/pretrained_model/model.safetensors` | `d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef` |

canonical 原始 receipt 为
`runs/canonical-furnace-received-20260914-002/runs/canonical-fullframes-20260914-001/checkpoint.json`。
其中恢复点只有 2500/5000，源代码也只登记这两个保存点。
检查过的本地恢复目录未发现独立 step-0 artifact；不是对整个远端存储的不存在证明。
历史 receipt 报告 345 个冻结张量未变；本次尚未重新逐张量验证该结论。

## 必须控制的接口差异

从两个实际 `config.json` 和处理器 JSON 确认：

| 项目 | 下载 base 默认值 | canonical 训练产物 |
| --- | --- | --- |
| action feature | 6 | 14 |
| n_action_steps | 50 | 1 |
| empty_cameras | 0 | 2 |
| scheduler warmup / decay | 1000 / 30000 | 16 / 5000 |
| load_vlm_weights | true | false（恢复完整权重） |
| 预处理 | 原始 processor、原始统计量 | top→camera1、Action Contract 投影、pi-ALOHA/有界夹爪转换、训练统计量 |
| tokenizer 文件身份 | 原始本地保存文件 | 训练保存文件 SHA 不同，chat template SHA 相同 |

这些差异不能自动解释为 bug。特别是两者 `observation.state` 的配置均留有
6 维占位；实际 ALOHA 状态合同必须取 dataset/adapter 的 14 维语义。
tokenizer 序列化 SHA 不同也不等于 token ID 一定不同，需要对固定指令实际核对。

主要科学对照应为：**base 权重 + 已登记 ALOHA 配置/处理器**，对比
**训练后权重 + 同一配置/处理器**。这测量该任务接口下的训练影响，不能称为
原始下载包默认配置直接部署的成功率。若另测原生默认接口，应另立语义兼容性实验。

重建零更新初始化时使用固定 upstream/base revision、历史 seed 和配置，
审计 missing/unexpected keys、全部参数/缓冲区、dtype 转换与冻结范围。
不得静默 `strict=False`、截断动作或替换随机初始化参数来制造对齐。
如存在加载时重初始化或无法复原的转换，单独报告，不能冒称历史 step 0。
保存新的零更新 artifact 并进行独立 reload；不得覆盖下载缓存或旧恢复点。

## 下一阶段比较规格（设计，尚不可直接启动）

比较固定三对：base→2500、base→5000、2500→5000。base→零更新 artifact
另外承担初始化/导出检查，不把重建 artifact 当成保存过的历史证据。

1. **全部权重和缓冲区**：按精确名称匹配，列出缺失/多余项、shape、原始
   dtype、元素数、逐张量字节 SHA、有限性、是否完全相同、变化元素比例、
   均值/标准差/RMS、L2 范数、差值 RMS/最大绝对差、相对 L2 和余弦相似度。
   零范数的相对量/余弦记为未定义；不同 dtype 的数值转换与字节差异分开报告。
   按 VLM vision/language、expert 各层 Q/K/V/O、MLP、norm、state/action/time
   projections 汇总，同时保留每个参数的行，不只给 top-K。
   端点差值是累计净变化，不能反推每一步更新幅度或优化器轨迹。
2. **固定输入推理 trace**：先一个输入做 plain A/A、独立 module-hook 和
   TorchLens 隔离进程对照，验证完整输出、选定激活、权重/缓冲区与 RNG。
   通过后再扩展同一 train/dev 样本表（hidden 不访问）。保存实际图像、状态、
   token IDs、camera mask、完整 50×32 初始噪声的身份及实际时间序列；
   不能只记录 seed 或截成 14 维。采集每个 denoise 调用的输入/输出、prefix、
   expert 投影和最终完整动作，同时分离标准空间、内部空间与安全投影后值。
   自由积分后各检查点的 x_t 不再相同，需明确这是累积行为差异；若比较同一点
   向量场，另行固定完全相同的 x_t 和 t，不能混淆两个问题。
3. **固定监督 forward/backward**：第一批限定已登记训练样本和固定
   noise/timestep、target/padding、train/eval 模式、BF16/AMP；验证真实 loss
   reduction 和每维有效分母。记录每个可训练参数梯度的 finite、norm、zero 比例
   和前后余弦；区分 frozen、grad=None、真零梯度。仅 backward，不构造更新流程，
   optimizer/scheduler step 均为 0；不将这次重算梯度称作历史训练梯度。
4. **Gate 对照**：完成语义、独立 reload 和采集器兼容性检查后，按新身份登记
   base/训练后同协议 Gate 3→4，环境种子和 policy 噪声种子 1000–1004，
   最多 500 步、原成功及全部安全阈值。各自通过 Gate 3 才进入 Gate 4。
   直接 Gate 主路径不插入未获接受的 TorchLens；中间激活优先离线重放。
   第一观测之后两模型状态会分离，闭环各自轨迹不能作为“同输入”激活对照。

TorchLens 2.23.0 的本地合成结果已表明 Python RNG bookkeeping 不透明，
且已有文档保留 Python-random 计算与图完整性的限制。每种真实 forward/backward
路径都要重新检查，不能由合成桥接测试推定 CUDA/SmolVLA 兼容。
失败保存 incomplete 和原始产物，不放宽阈值；必要时以透明 hooks 作为正式数值
参照，TorchLens 仅保留未接受的辅助图证据。

## 资源、停止条件与接续

下一次本地权重代数拟使用现有固定 Docker image
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`，
WSL Bash 启动、无网络、只读输入、2 CPU、4 GiB 且无额外 swap、逐张量/分块计算，
每阶段最多 600 秒。运行前实测 cgroup、可用空间，并绑定新脚本/计划 SHA；
不同时构造三份完整 policy，不做全模型 SVD。当前只有静态检查脚本，数值比较
及真实 trace runner 仍需实现和验证，本文件不是已就绪的调度计划。

Docker 恢复前继续静态准备，不使用 Windows/WSL 主机 Python 绕过环境规则。
本轮的 WSL 沙箱访问错误已通过正式权限流程完成只读重试；随后确认的是 Docker
手动暂停，并非自动审核拒绝，也不是缺少模型文件。

需要真实 CUDA trace 时再请用户提供当前 SSH 窗口。届时先只读 doctor/缓存/
资源检查，冻结样本、代码与数据身份、forward/backward 次数、期限、输出和停止
条件，再在 content-addressed workspace 中执行零优化器诊断。完成后回传并核验，
按 AutoDL 约定关机并确认平台状态；不释放实例。尚未登记具体 CUDA 预算，
不应提前开 GPU 等待本地实现。只有初始化身份无法恢复或旧产物无法回答问题时
才评估重训，并先给出独立小规模计划。

立即停止条件：identity/schema/输入不匹配、冻结参数意外变化、非 finite、
采集器输出或 RNG 不符合该阶段合同、超时/OOM/资源超限。失败不自动重试旧 run。
历史 canonical 004/005 仍为 Gate 4 0/5；原始模型 Gate 4 未测。
现在既不能断言“训练破坏了原本能成功的策略”，也不能由权重差异定位唯一根因。
