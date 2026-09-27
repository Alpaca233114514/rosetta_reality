# Pre/post 远端无卡前置核验

当前 SSH 窗口完成只读身份检查，未启动模型、训练、仿真或 GPU。
证据：`.cache/prepost-remote-preflight-001/result.json`；检查器
`.work/prepost-remote-preflight-20260923-001.py` 仅使用 Python 标准库，
流式读取文件 SHA 和 safetensors JSON 文件头，没有读取数值为张量。

## 实测状态

- 平台匹配既有目标实例，运行中、无卡模式；`nvidia-smi` 返回无设备。
- cgroup 实际限制：内存 2,147,483,648 字节，额外 swap 0，
  CPU quota/period = 50000/100000，即 0.5 CPU。
  主机 `free` 显示的数百 GiB 不代表容器可用内存，不能据此加载模型。
- 系统盘 30 GiB，约 26 GiB 已用、4.4 GiB 可用；数据盘 50 GiB，
  约 45 GiB 已用、5.3 GiB 可用。该快照早于本轮归档/清理。
- Python 3.12.3；torch 2.8.0+cu128；LeRobot 0.6.2；transformers 5.5.4；
  safetensors 0.8.0；Accelerate 1.14.0。TorchLens 未安装。
  版本来自 distribution metadata，没有导入 torch。
- `configs/runtime/autodl_rtx4090.yaml` SHA 为
  `be2bfc3ea2a518c85e56410ba3ea1da6f744236d51b6f4a5f6a7b73927e9f992`。
  当前无 GPU，因此不宣称通过 CUDA doctor 或 profile GPU 门禁。

## 身份核验

远端三个完整权重重新流式 SHA256，与本地已验收身份一致：

| 对象 | SHA256 |
| --- | --- |
| pinned base | `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb` |
| canonical 2500 | `a3494600183d0a24c71ab7db0976afbc57567ac714ac678fd572c607e06a2f0c` |
| canonical 5000 | `d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef` |

既有源工作区 archive marker 为
`6bb3da3027dfc66fe876551977cdbaedd2f217f877dc4386f7f4269db538c96e`。
marker 是记录值，本轮没有重建全部历史 archive，所以不冒称当前全工作区重验。
另外实算六个关键 LeRobot 源文件 SHA，完整值见机器证据；其中
`modeling_smolvla.py` 为
`37b1d56f37510732a087cf5c32c05cd15d6234201a3f002f108ec4c53438cc7d`，
`smolvlm_with_expert.py` 为
`996d3b0c713c0ed42b383aa2cf89b2e6f9868e337747c480a64adaecdc1073cf`。
后者与本地用于静态阅读的源码 SHA 相同。

两个 optimizer 文件头包含 BF16 的一阶/二阶矩状态。文件头记录的索引
尚未与运行时每个参数名逐项绑定，不能据它们断言某个特定 norm 有非零历史梯度。
完整诊断还需实际参数顺序/优化器映射、固定样本梯度和有效更新精度证据。

## 连接排查与边界

最初 WSL 调用与 Linux SSH 未返回结果，已停止有界尝试。
最终从 Bash 使用现有系统 OpenSSH 成功完成相同只读操作，未修改 SSH
配置、凭证、安全设置或远端服务。平台 JupyterLab 被短暂打开用于排查，
没有在其中执行命令；用户指出不需要后关闭该页面，后续均使用 SSH。

全参数数值比较与独立复算已在本地固定 Docker 完成，见
`reports/training/m2-smolvla-prepost-parameters-result-2026-09-23.md`。
本轮无卡检查不能替代同输入 forward/backward、零更新 artifact reload、
TorchLens 兼容性或原始模型 Gate 3/4。没有为这些未测阶段临时改用主机
Python、无控 CPU/offload 或扩大内存。

归档任务由用户指定的 Sol / medium 子代理独立执行；其上传、内容校验、
删除及最终关机状态应以本轮归档收尾记录为准，本报告不预先声明完成。
