# Iris 新炉准备登记

新名称 `iris-k-scene-20260913-001`，分支 `codex/iris-k-scene-20260913`。
用户于本轮明确授权新炉训练、新分支和腾空间。现有工作与历史证据保留。

本记录仍为 `launchable=false`；训练授权成立不代表前置核验已通过。
沿用 `m2-smolvla-hestia-k-scene-regularization-design-2026-09-13.{md,json}`
的单轴设计：新鲜基座两臂，lambda 0 / 0.01，batch4，各1280更新，
seed20260809，warmup16/cosine1280，保留320/640/960/1280。
两臂各自隔离的两步 smoke 不进入正式权重或优化器状态。
总训练预算2564个更新；开发集合仍为开发集合，hidden封存。
正式阶段必须另有封存的源码、校准、配置和阶段验收，不自动执行本草案。

## 先行数值核验

`iris-attention-backend-20260913-001`只重算上一轮320个已封存attention
张量样本，不加载policy、不读原始训练样本、不执行optimizer。
输入为既有Q重放workspace中的 `runs/hestia-q-replay-20260913-001/`。
脚本 `scripts/audit_iris_attention_backend.py` 要求1920个输入张量哈希匹配
原CUDA事件，重算概率和输出与原记录完全相等，任何差异均记录失败。
每个输入文件SHA写入新输出；保留原float64/BF16一ULP失败，不放宽阈值。
这是不同后端算术合同的定位补充，不把原失败改为通过。

通过 `scripts/stage_autodl_from_wsl.sh` 创建新内容寻址workspace后，使用
`configs/runtime/autodl_rtx4090.yaml` 和 `scripts/run_autodl.sh preflight`
先 doctor/benchmark 再运行。单次算术worker限180秒、CPU RSS上限4GiB，
预计不到3分钟；输出create-only，失败不重试、不自动进入训练。
CUDA空闲和登记实例身份已经现场核对；数据盘剩余约4.9GiB，完整两臂
恢复checkpoint预算尚未满足。空间处理须保留独有模型、数据和失败证据。

## 工程修复范围

默认关闭的v2 feature、原生loss之后的标量正则、分项记录、零系数完全旁路、
异常后恢复、校准身份校验和新run隔离。未验证的正式resume继续禁止；
不把scheduler、额外loss、V回滚、dropout或unfreeze混入本单轴。
训练效果、独立reload、Gate3/4均为 `not measured`，M2未完成。
