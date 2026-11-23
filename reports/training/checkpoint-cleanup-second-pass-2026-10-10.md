# Checkpoint 第二轮清理 — 2026-10-10

用户要求再次检查 C、D 盘并删除重复或无用途的 checkpoint，明确保留正式结果与日志。本轮额外删除一个 Iris 重复回传包，释放 **2,414,888,960 bytes（2.25 GiB）**；C 盘没有新增删除对象。前一轮本地 51.22 GiB 加本轮共约 **53.47 GiB**，云盘本轮未操作。

删除对象为 `runs/iris-recovered-20260913-002/archive.tar`，实际位于项目 D 盘存储。删除前两次检查全部 126 个普通文件成员，与同目录 `verified/` 内文件的大小和完整内容逐字节一致；包括两个完整 policy 导出、processor、日志和 manifest。归档成员 payload 为 2,414,660,775 bytes，旧报告的 125 个 payload 文件计数不包含归档内 manifest；此次计数包含 manifest。正式解包结果和全部日志保留。删除后精确路径已不存在，126 个对应成员全部存在且大小未变；保留目录实际有 127 个文件，多出原有的 297-byte `transfer-receipt.json`，该收据也保留。

复查范围包括项目 C 盘 `.work`、`.cache` 和基础模型目录、项目 D 盘存储及其子目录，以及用户文档、下载、桌面、工作树、用户缓存与 Temp 内的 checkpoint 文件名。未发现其他可确定淘汰的大型 checkpoint；不能据此断言全部文件内容互不重复。D 盘发现的项目同名异常目录为空。本轮没有全盘扫描系统文件、清理回收站、修改其他项目或检查/压缩 WSL、Docker 内部存储。WSL distribution 只读枚举返回 `Wsl/E_ACCESSDENIED`，没有尝试改变服务、权限或环境。

继续保留的主要对象：

- C 盘三个 A/B/C `005000` 完整终点，以及固定 revision 的 SmolVLA 基础模型。
- Hestia 四点、Iris 两臂各四点、canonical 2500/5000 和正式 Gate/历史导出。全部 31 个剩余项目大型 `model.safetensors` 文件存在且大小与本轮初始盘点相同；这里不是模型 reload 或完整权重身份验证。
- 两个 coverage40 256 步 checkpoint。`m2-smolvla-hestia-checkpoint-retrieval-result-2026-09-11.md` 明确指出旧 control 的模型、optimizer、RNG 与正式 B 不同，不能凭相近目录名认定重复。
- `runs/autodl/prometheus-20260813/prometheus-evidence-20260813.tar.gz`：41 个成员包含正式权重，目前没有对应本地解包成员，不能删除它而丢失本地正式结果。
- vfunfreeze 的不完整接收权重：没有核验到完整本地替代文件，本轮继续保留；不因 `.incoming` 名称自动删除。

逐字节核验只针对拟删除的重复 Iris 归档，不对所有权重做无目的 hash 扫描。未加载模型、启动 ML/训练/仿真、连接 AutoDL、提交或推送；用户既有源码修改及科学结论保持。

证据：`.work/checkpoint-cleanup-20261010-002/iris-archive-verification.json`、`local-receipt.json`、`summary.json`；第二轮候选盘点及归档成员列表为 `.work/checkpoint-cleanup-20261010-002-*.json`。收据中的 `storage/` 表示项目 D 盘存储根的逻辑别名。
