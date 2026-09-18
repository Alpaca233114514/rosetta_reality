# 闭环复现性诊断：本地实现与轻量验收

分支：`codex/closed-loop-reproducibility-20260918`。
本轮仅代码、静态检查、小张量/假环境测试和极小 MuJoCo 状态读取。
没有真实模型或数据加载、GPU、ALOHA rollout、SSH、训练、提交或推送。
正式 Gate 4 保持 0/5，M2 未完成。

## 交付

- `collect-repro`：独立注册的 baseline_a、baseline_b、full_trace 三臂采集。
  保留原噪声与动作，记录完整模型边界和 MuJoCo 积分状态、模型与运行身份。
- `compare-repro`：先验明身份，再比较 A/A 与 A/full_trace，定位首次逐位分叉。
  dtype、终止条件、逐步连续性、证据 SHA 和 trace 对账均纳入验证。
- 原始 Gym observation 中的 NumPy 数组转为独立张量副本，避免采集序列化失败。
- 新不可执行草稿和文档：`configs/vla/closed_loop_reproducibility_001.json`、
  `docs/m2-smolvla-closed-loop-reproducibility.md`。

## 验收

Windows 通过 WSL Bash 启动既有固定 Linux Docker 镜像：
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`。
网络关闭、无 GPU、源码只读；pytest 限 2 CPU / 3 GiB，静态检查限 1 CPU / 512 MiB。

首轮 53 项通过；补充 NumPy、重新封 SHA 后丢步骤、环境身份漂移及旧包拒绝反例后，
最终 **57 passed，0 failed，0 skipped，18.76 秒**。XPU 零设备提示仅为导入警告。
Ruff、`git diff --check` 和新 draft schema 检查通过；execution_ready=false。
格式整理最初因长行及只读 Ruff 缓存失败，改为副本格式化、no-cache 后通过，
没有放宽源码挂载保护。原始首轮与最终 pytest 日志分别保留。

证据：`runs/closed-loop-reproducibility-local-20260918-001/` 中的
`pytest.log/xml`、`pytest-final.log/xml`、`ruff-final.log`、`draft-schema.json`。

## 未验证

真实 CUDA、实际 dm_control/ALOHA 采集、三臂真实闭环等价、完全无插桩等价和
任意中间物理状态恢复均未验证。两基线都有相同观测开销；积分状态不包括所有
Python wrapper/RNG/callback 状态。后续 SSH 或 GPU 验证前按用户要求先通知用户，
获得当前窗口授权并使用新注册，不能直接执行 draft。
