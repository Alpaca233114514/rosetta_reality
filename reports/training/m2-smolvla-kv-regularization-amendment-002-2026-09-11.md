# 正则化诊断入口修正 002

原计划的 31 项 synthetic/regression checks、Ruff 和格式检查均通过。
实际 CLI 在读取计划/特征前报错：

```text
ModuleNotFoundError: No module named 'rosetta_reality.vla.contextual_kv_probe'
```

pytest 的项目配置提供 `src` 路径；直接运行脚本没有相同环境，访问到了镜像
已安装包。原失败日志保留在 `runs/kv-regularization-preflight-20260911-001/`，
未创建研究结果目录，不作为正则化假设的负结果。

修正只在 runner 显式设置 `PYTHONPATH=/workspace/src:/workspace`，并在读取特征
前核验模块 `__file__` 位于当前源码挂载路径。算法/代码、网格、预算、数据、
折分、指标及阈值完全不变；研究输出改用独立 `-002` 目录。新 JSON 继承并
校验原计划 SHA，原文件不覆盖。重新运行静态/合成预检，然后执行一次分析。
