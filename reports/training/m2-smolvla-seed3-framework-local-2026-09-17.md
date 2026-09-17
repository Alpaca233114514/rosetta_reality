# Seed 3 诊断框架：本地实现与合成验收

已交付固定 canonical step-5000 的独立 Seed 3 入口及可解释性框架。
**这是代码与合成测试验收，不是真实 Seed 3 rollout 或训练结果。**
正式 G4 seeds 1000–1004、阈值、历史证据和 M2 状态均未改变。

## 实现

- 统一 CLI：`scripts/diagnose_smolvla_gate.py`，包含 validate-plan、collect、
  replay、probe、audit-training、analyze、verify 七个命令。
- 新增 `src/rosetta_reality/eval/gate_diagnostic_*.py`：只观察原在线调用，
  保存原始输入、实际 batch/完整噪声、完整输出、内部动作、decoder/projection，
  复用原 rollout trace；独立 NumPy 校验器检查文件、控制、单变量身份和分组算术。
- 最多 12 时点、60 次探针，先精确重放/自拷贝控制，随后单独替换 image/state/noise；
  干预输出不送入仿真。缺失物理/历史梯度保持未验证，失败证据不覆盖。
- 训练审计绑定历史模型、训练计划、源码及实际操作数；旧摘要缺少绑定时标记
  insufficient_evidence，不把旧 passed 当作当前训练代码正确性的证明。
- `eval.action_metrics` 改为延迟导入并保持公开函数身份，使文件-only 校验不导入
  torch、lerobot 或 MuJoCo。原 Gate、在线适配器、processor 和仿真器源码 SHA
  与 traced Gate 005 清单逐项一致；新注册不能重封修改后的旧 Gate 绕过前置验收。
- 新 draft：`configs/vla/gate_seed3_diagnostic_001.json`；执行配置为空，不能启动模型。
  使用说明：`docs/m2-smolvla-seed3-diagnostics.md`，架构导航已同步。

## 实际验证

通过 WSL Bash 调用 Docker CLI，在现有固定 Linux 镜像运行；网络关闭、CPU 2、
memory 与 memory+swap 均 3 GiB，测试时源码只读，仅结果目录可写。
镜像：`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`。
Python 3.12.3，PyTorch 2.11.0+xpu；本次没有 CUDA/XPU 设备暴露给容器。

最终结果：**109 passed，1 skipped，0 failures/errors**。跳过的是需要真实 CUDA
的像素一致性检查；一个 XPU 零设备提示不代表执行了 XPU 推理。
Ruff 和 `git diff --check` 通过。

测试范围：新增框架、原 rollout trace、action metrics、动作/processor 合同、
图像缩放与 canonical posttrain identity。新增测试包含：

- 合成采集与未采集的完整 noise/action/state/reward/终止条件等价。
- 相机错误、重复缩放、噪声错配、动作维序错误的首个边界定位。
- padding、loss 分母、optimizer 参数遗漏、scheduler 计数反例。
- 源码/权重身份漂移、旧 Gate 重新封装拒绝、输出路径越界与覆盖拒绝。
- 环境异常、写盘失败、参数变动、证据预算、无物理信息和不完整任务状态。
- 原样/自拷贝控制、BF16 精确存储、独立分组数值复算、重新封 SHA 后仍拒绝
  篡改指标、丢记录、维度顺序变化和错误跳过 donor。
- 单步无 donor 明确跳过，以及 12 时点恰好 60 次 probe 上限。

初次合成运行有 25 passed/1 failed：无模型导入测试发现 eval 包的 eager metrics
导入间接初始化 torch；延迟导入修复后复测通过。初次只读 Ruff 缓存写入失败，
改为 `--no-cache`，没有改变 Docker 保护或科学门禁。早期边界回归 69 项通过，
扩充后 106 passed/1 skipped，再加入历史引擎身份和最大探针预算回归得到最终结果。

## 可查看的合成示例

证据根：`runs/gate-seed3-framework-local-20260917-001/`。

- `validation-final.log`、`pytest-final.xml`：最终检查和合成示例校验输出。
- `pytest.log`、`pytest.xml`：前一轮 106 项通过的结果，保留未覆盖。
- `synthetic_demo.py`：用 tiny CPU 张量与 fake environment 生成示例的脚本。
- `synthetic/collect`：13 步假环境采集；`synthetic/replay`：逐步完整重放。
- `synthetic/probe`：12 时点、60 次合成 forward，零 optimizer。
- `synthetic/training-audit`：核验现有两份历史摘要的文件身份，明确记录缺少绑定。
- `synthetic/analysis/report.md`、`result.json`、`timeline.csv`：解释报告格式。

五个示例阶段都完成文件/结构校验；collect/replay/probe 还完成对应独立数值校验。
报告标记 provenance_kind=synthetic，不能把假环境的 reward/success 当作模型表现。

## 未验证与边界

本轮没有下载、读取真实模型权重/样本、真实 MuJoCo rollout、CUDA forward、
optimizer 更新、远程操作、训练、checkpoint 选择、Git 提交或推送。
真实 CUDA 重放、traced/untraced 轨迹等价性和 Seed 3 任务结果保持 **not measured**。
后续真实采集仍需新登记的运行身份、当前授权窗口、匹配的 artifact/G3、运行环境和
watchdog；本实现及合成检查不构成实际任务验收或唯一根因结论。
