# SmolVLA 本地恢复清点 — 2026-09-10

**已找到历史源码生成片段和结果摘要，尚未找到旧 8 场景 checkpoint。**
这份清点不证明远端最终代码已恢复，也不授权新训练。M2 未完成。

## 找到的本地证据

`runs/smolvla-visual-repair-20260909-003/` 保留了上一轮控制脚本和日志。
同目录的 `remote-native-report-001.log` 可解析出历史汇总；
`remote-native-final-audit-001.log` 和 `remote-native-finalize-001.log` 记录
47 项回归通过，但本轮没有重跑这些测试。原始日志保持原位，不直接发布。

| 历史 cyclic 协议 | 正确图 MSE | 错配图 MSE | 视觉无关均值下限 | 正视觉差值 |
|---|---:|---:|---:|---:|
| train 8 | 0.0263029864 | 0.3753699440 | 0.1926252405 | 4/4 |
| dev 5 | 0.2173488564 | 0.1847120401 | 0.1214306170 | 0/4 |

本地日志与交接的负结果一致；不是新的模型测量。新 all-pairs 指标尚未测量。
日志记录 pilot 约 131.73 秒、峰值 CUDA allocation 2,817,605,120 字节；
这些数字属于旧实例，不能作为当前 worker 的实测资源预算。

清单 `reports/training/visual-native-recovery-20260910/manifest.json`
保存 19 个本地小文件的 SHA-256、来源关系、搜索范围和缺失项。
其中 5 份 `.py.txt` 是从历史 shell heredoc 提取的完整源码片段；只统一
换行为 LF，不执行、不安装到 `src/` 或 `scripts/`。它们早于远端 Ruff
格式化，尚未与 `final-audit.json` 的最终文件哈希对齐。

| 恢复对象 | 本地来源 | 当前可核验范围 |
|---|---|---|
| 跨 episode 固定采样与对应测试 | `remote-native-edit.sh` | helper 源码、split/scope/index 防护；feature 修改配方 |
| preflight 使用解析后的 batch/episodes | `remote-native-edit.sh` → `remote-native-register.sh` | batch 来源及数据构建前的 train-split 检查顺序 |
| Trackio 继承配置与异常清理 | `remote-native-trackio-fix.sh` | resolved context/plan 合成、保留主异常的修改配方及测试 |
| 失败边界集成回归 | `remote-native-finalize.sh` | formatter 之前的完整测试源码 |
| 旧 cyclic 评估 | `remote-native-evaluator.sh` | 历史 evaluator 源码；不得当作新的 all-pairs 实现 |
| 旧 plan 迭代与 checker 修订 | `remote-native-register.sh` → `remote-native-trackio-fix.sh` → `remote-native-retry.sh` → `remote-native-continue.sh` | 创建顺序和修改意图，缺少最终 resolved config |

表中脚本均相对于上述本地控制目录。恢复现有文件时先检查每个替换的旧
文本唯一匹配，再在新工作副本应用；不得直接运行这些控制脚本。它们包含
历史训练、目录创建、tmux 和电源操作，执行授权不能由“文件存在”推导。

## checkpoint 与远端身份

按文件名清点 `runs/`、`artifacts/`、`checkpoints/`（后者的 junction
目标也已检查），排除 `runs/compiler_cache/`，共 10,971 个文件。
没有路径匹配本轮 native pilot 身份；所检查的常见权重文件名共 78 个。
这是指定根目录的清点，不声称搜索过所有磁盘或所有可能的命名方式。

9 月 9 日出现的本地 `model.safetensors` 仅 52,199,424 字节，路径属于
vfunfreeze step-632 的 `.incoming-...` 传输目录；对应日志记录传输失败。
它既不是 native 8 场景控制，也没有完成本轮完整性核验，不能替代旧 A。

用户新提供的 worker 经只读检查后确认是另一台老实例。其数据盘未找到
`dev/visual-utilization-20260909-001` 或本轮 native 结果；SSH 已退出。
GPU 元数据为 RTX 4090 D、24,564 MiB；登记环境包元数据为 Python 3.12.3、
Torch 2.8.0+cu128、LeRobot 0.6.2、Trackio 0.28.0。这不是模型预检通过，
没有加载模型/数据，没有运行训练，也未验证平台当前计费状态。

在授权推送前，成功的 `git ls-remote` 显示 GitHub 尚无匹配 visual/vision
的功能分支，main 为 `14f320d981e7fdc53142a314e6d4cc9f3ea58940`，仍在 M1。
本地 HEAD 比它领先 41 个提交。老 worker 上的 Git 查询超时（exit 124）
没有被当作“分支不存在”的依据。用户随后授权推送**当前功能分支**。

## 可继续与必须补齐的内容

可以提交本地源码、历史实验文档、本轮 8→40 计划和这份恢复清点，保留
当前分支历史。权重、数据、原始运行目录和主机连接信息不进入这次提交。
源码片段也不会自动成为远端训练时的最终源码身份。

下一次新训练前仍需取得：

1. 原 native 8 场景 step-256 checkpoint 的位置、完整性清单和权重哈希。
2. 原实例最终 `final-audit.json`、未提交代码及对应最终文件 SHA-256。
3. 旧 `-003` plan、checkpoint 的 `train_config.json`、seed、optimizer/
   scheduler 全字段与 normalization/cache 身份。
4. 原报告和完整 eval/reload JSON；当前目标 worker 的真实预检。

优先在原实例取小型清单；需要迁移时安排服务器之间传输到新版本目录，
不经本机中转大文件。只有确认身份后，才把代码绑定到已授权提交或新
content-addressed workspace。新计算预算仍需单独授权；失败后停止。

本轮完成的是资料核对、计划、恢复清点和 Git 交付；视觉泛化仍未修复。
