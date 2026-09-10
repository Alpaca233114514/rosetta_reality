# Athena：CPU 更新观察器验证登记

状态：预登记，仅执行合成 CPU 检查；不加载 SmolVLA 权重、真实数据或历史 optimizer state，不启动训练。

延续修订草案第 5、6 项。已在原 A 实例完成的 scheduler/sampler 合同结果保留；本模块在保存 B 的已登记 AutoDL 平台容器执行。用户一次只能开启一台无卡实例，因此顺序核验，不要求两机同时在线。

- 作业名 `athena-observer-cpu-001`；源码为包含本登记的功能分支不可变 commit，新建对应 workspace，运行时记录确切 commit、文件 hash、环境与退出码。
- CPU 上限 600 秒：Ruff 30 秒、观察器测试 120 秒、checkpoint 流式 checksum 与空间清单 180 秒；超时或任一检查失败即停，保留日志。禁止下载、GPU、嵌套 Docker、自动依赖安装和 checkpoint 修改。
- 新模块 `src/rosetta_reality/vla/training/observation.py` 只包装 native cycle/update 接口，不修改已登记 trainer 文件；按实际交付批次和成功 optimizer step 分别计数，并记录实际 LR。先安装观察器，再安装 feature wrappers，按相反顺序恢复。
- 测试 `tests/test_smolvla_training_observation.py`：真实 CPU AdamW/LambdaLR，以及原生 `update_policy` + CPU Accelerator + 单参数合成 policy 的前后对照；权重、RNG、loss、LR 和 batch 身份必须相同。该合成 policy 不代表 SmolVLA forward 已通过。
- 验证缺批、多批、错序、重复、未知身份、预处理失败、更新前/后失败、skip、多次 step、没有 step、非同步累积与 wrapper 恢复。错误即使被上层捕获也不得伪报 complete。原生依赖测试跳过也视为预检未完成。
- B 的完整 checkpoint 只读列出文件大小、模型流式 SHA-256，并核验既有期望 hash。将六份完整 checkpoint（四个 quarter、一个 smoke、一个原子临时副本）与 2 GiB 余量和实际可用空间比较，不删除历史文件。

全部 CPU 检查通过仍不自动启动 Athena。B/C 新评分合同、真实预检、batch 1/4 forward、独立两步 smoke/reload 及磁盘均需通过后，才能冻结主诊断执行计划。GPU 时间、视觉改善、Athena 训练和 Gate 3/4 当前均为 `not measured`；M2 未完成。
