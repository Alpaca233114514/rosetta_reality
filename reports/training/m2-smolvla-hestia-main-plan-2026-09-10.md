# Hestia：主执行链预登记与 SSH 前交接

状态：代码准备完成，**本版本 Linux Ruff/CPU 回归和 GPU 主实验均未执行**。用户要求本轮做到 SSH 门前后结束，不连接或要求开启实例。上轮已测的 `4098c2b` / 执行源码 `e289f45` 的 184 项回归不覆盖本次新主执行链。

本轮仅完成静态源码审阅、100 字符行长检查、`git diff --check` 和待执行启动脚本的 `bash -n` 语法检查。新增 13 个主链反例测试已写入 Linux 验收清单，尚未运行；不能称为已通过 197 项回归。源码未运行于 Windows Python，也未连接 SSH。

## 科学问题与唯一干预

历史 8→40 场景固定 256 步的结果仍为负：A/B 开发均未通过，B 训练关节 4/4、夹爪 1/4。Hestia 检验扩大原生拟合预算能否改善同一批开发场景。C 使用 40 个 frame-0 场景、batch 4、seed 20260809、1280 updates；cosine decay 同步从 256 改为 1280，warmup 保持 16。**这是更新预算与学习率轨迹的联合干预，不能解释成纯重复次数效应。**

基座/VLM/数据/LeRobot revision、processor、Action Contract、split、train-only normalization、原生损失、VLM 冻结范围、expert/state projection、AdamW 及其余设置均沿用已登记合同。候选名 `m2-smolvla450m-visual-hestia-fit40-001`；使用新基座加载与新 optimizer，C 目标目录已存在即停止。B 复用原 checkpoint，不重训、不续跑其 optimizer。没有 paired loss、dropout 或 unfreeze。

官方 SmolVLA 资料说明架构与训练流程；本项目“增加拟合预算可能修复开发视觉利用”是待检验假设，官方文献未证明这一具体修复。此前官方资料核对及覆盖实验依据见 `m2-smolvla-native-visual-coverage40-plan-2026-09-10.md`，不以本实现替代实验证据。

## 入口、身份与执行顺序

入口 `scripts/run_hestia_fit.py`，运行目录 `runs/hestia-fit40-001/`。`visual_fit_job.py` 负责启动前证据、PID 启动时刻和历史 B 全数组复现校验；`check_hestia_collector_cpu.py` 增加新主执行链及其反例测试。

1. 在新 clean detached checkout 使用 `scripts/run_autodl.sh shell`；registration 封存全部 Git 文件 SHA、commit、计划、唯一炉名、1800 秒预算和执行/关机授权。watchdog 独立于 SSH，绑定 PID、进程启动时刻及 registration SHA。只使用已登记 AutoDL Linux 容器，不嵌套 Docker、不下载模型/数据。
2. 按现有 SHA 恢复小型前置报告和 B 计划；校验 durable train-view 链接。先运行新版 Ruff、CPU 回归、5120 合成 sampler 序列及 B 实际配置/恢复状态检查。失败不进入模型阶段，不能把上轮 184 项通过移植给新版本。
3. 核验已完成两步 smoke 的原报告、模型和 reload 元数据，以及其相关原生训练源码、已登记运行依赖。只复用已验证的两步能力证据，不复用 smoke optimizer 或权重作为 C 初始化。GPU 型号要求与原预检相同的一张 4090 D，UUID 按当前设备重新记录；前置源码或环境不一致即停止，不能静默重做 smoke 或降低门槛。
4. 当前 `check_env`、doctor、真实缓存 data test（无 skipped）、benchmark、batch 1/4 的 no-optimizer forward、45 个非 hidden frame-0 的相机/状态/动作/train-view 合同检查。样本合同检查使用原 B，C 尚不存在。此前报告不是当前平台身份凭证。
5. 封存独立 training permit，绑定前置结果、完整 sampler 顺序、计划和活跃 watchdog。训练前还需剩余至少 1200 秒并重验磁盘。随后使用原 guarded v2 launcher 的 bounded diagnostic `smoke` 入口跑 C 的 1280 步，保留原生 loop；观察器记录实际交付/成功消费 5120 样本、1280 个成功 optimizer step、完整 LR，另检查每次 step 的 finite gradient。
6. 保存 320/640/960/1280 四个完整恢复 checkpoint；逐 tensor 验证 frozen VLM 不变、expert/projector 有更新、只有精确等值 BF16→FP32 转换可作为序列化差异。封存每个 quarter 全文件清单、scheduler/step、finite metric、实际 saved config、processor 与 normalization。
7. C 完整性通过后才封存 B/C 模型采集合同。固定末步 1280，使用其原生 pretrained artifact 原位导出清单，不复制大权重、不做 checkpoint 搜索；先记录 reload 待验证，最终状态以独立 reload 结果为准。
8. B 和 C 各自两次独立进程采集七组完整数组，保持相同图像、状态/语言、四组噪声与 50-step chunk。所有数组 dtype/值必须完全一致。新 B 还需与历史 Hermes B-first 的 hash-bound manifest 和七数组逐项复现；不改历史负结果或只比首动作/聚合指标。
9. 完整证据通过后写固定 B/C 开发比较。科学门槛未通过时写 `negative_result`，保留全部证据并结束；工程失败写失败阶段并结束。二者均不触发重试、续跑或下一炉。

完整 B/C collector 按既有合同要求 C 证据齐全，因此历史 B 的全数组复现在训练后完成；其真实 checkpoint/processor/config 完整性与非视觉条件已在训练前检查。若训练后复现失败，该炉不能解释为 C 带来的效果。

## 预算、保留与停止条件

- 一个 supervisor 总计 1800 秒，含 CPU 验收、真实预检、主训练、封存、四次采集和比较；CPU 验收上限 600 秒，内部单命令 180 秒，主训练单阶段上限 900 秒。所有子阶段服从同一个剩余期限；不自动续费或重开。
- 时间依据为历史 B 的 256 步约 137.17 秒，先按五倍约 686 秒估算 1280 步，增加保存与完整评估余量；这是预算推估，非新硬件/新炉实测。已完成 smoke 每次 180-case reload 约 66.5 秒，四次采集预计数分钟。
- allocated ≤8 GiB、reserved ≤10 GiB、host RSS ≤10 GiB；资源守护不改变 batch、dtype、冻结范围或 scheduler。非 finite、OOM、身份/合同错误、期限不足立即停止。
- 四个新 checkpoint 加一个写入临时副本、另留 2 GiB。上轮按 B 实际完整大小估算需 10,217,718,298 bytes，约余 760 MB；本轮启动前必须重测，不以旧余量当作当前事实，不删除历史 checkpoint。
- worker/子进程退出后留 120 秒小证据回收窗口，然后调用已核验的关机 helper；若检测到其他任务或 helper/Trash 状态异常则保留失败记录，不擅自清理或释放实例。平台电源/计费状态需单独核实，不能只从 SSH 断线推定。
- 稳定训练只在独立控制 shell 每五分钟最小采样一次；不向训练进程注入 sleep，不因日志安静而杀任务。失败/quarter/结束时检查完整证据。

## 验收与解释

C 的 train40 和 dev5 都要求关节、夹爪、aggregate 在相同至少 3/4 噪声条件下，正确图误差低于全非自身错配平均误差及视觉无关均值下限。保持相对 B 的开发正确图误差和视觉增益改善，以及 standard full-chunk/first-action 关节/夹爪非回退；不能先平均错配预测。B 的已知夹爪负结果保留，前提仍是其原 aggregate 训练拟合通过。

五个开发 episodes 已被使用，不是独立测试；hidden split 保持封存。即使离线门槛通过，仍未证明注册闭环任务成功，M2 保持未完成。后续 Gate 3/4 需要单独处理，不由本脚本自动启动。

## 下一次 SSH 后的待执行命令

下列命令**本轮未执行**。先在获准服务器通过已推送功能分支建立新的 detached checkout，核对交接记录中的 commit；不要在旧 workspace 上 pull 覆盖修改。`ROSETTA_AUTODL_ROOT` 指向已登记 durable root。

```bash
# 在新源码 checkout 内；解释器使用该实例已登记的 smolvla-cuda-001 环境。
export PATH="$ROSETTA_AUTODL_ROOT/envs/smolvla-cuda-001/bin:$PATH"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runs/hestia-dispatch-main-001
set -C
cat > runs/hestia-dispatch-main-001/start.sh <<'START'
set -euo pipefail
bash scripts/run_autodl.sh shell <<'INNER'
set -euo pipefail
exec python scripts/run_hestia_fit.py supervise --execute-authorized --shutdown-authorized
INNER
START
nohup bash runs/hestia-dispatch-main-001/start.sh \
  > runs/hestia-dispatch-main-001/controller.log 2>&1 < /dev/null &
```

恢复前仍待现场核验：SSH 对应实例、GPU、当前软件/模型/缓存身份、B/旧 smoke/Hermes bundle 的存在与 SHA、磁盘、无其他活跃任务、新主执行链 Linux Ruff/回归。代码和上述命令仅为待执行准备，不构成本次远端执行记录。
