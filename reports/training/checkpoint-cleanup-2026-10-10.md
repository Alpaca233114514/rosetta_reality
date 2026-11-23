# 探索性 checkpoint 清理记录 — 2026-10-10

用户明确授权淘汰本地和 Google Drive 中探索性、无保留用途的 checkpoint，随后明确包括 C 盘。此次是淘汰产物，不是将所有 checkpoint 迁移到云盘后删除本地副本。

| 位置 | 已删除文件数 | 文件字节合计 | GiB |
| --- | ---: | ---: | ---: |
| C 盘项目 `.work` / `.cache` | 50 | 27,531,215,272 | 25.64 |
| D 盘项目存储 | 43 | 27,471,406,260 | 25.58 |
| Google Drive | 213 | 7,396,210,383 | 6.89 |

本地总计释放 51.22 GiB 的文件数据；云盘按已删除文件大小统计 6.89 GiB，没有读取 Google 存储配额账单。完成时文件系统可用空间约 C 盘 181.46 GiB、D 盘 107.68 GiB。

本地删除对象包括：探索性 A/B/C 的 1250/2500/3750 中间权重与 optimizer/RNG、展开的 `last` 重复权重、C 盘已有完整 D 盘版本的接收残片、已完成工程用途的 Hestia/coverage/Iris/训练链/paired 两步 smoke 与 Zen 单步 smoke，以及相应传输包和 Odyssey 重复导出权重。保留配置、processor、日志、manifest、恢复索引与失败记录。淘汰的训练状态不能再从已删副本恢复；历史 manifest 没有改写，不能再凭其旧完整状态认定文件当前存在。

保留并回读确认：A/B/C 三个 `005000` 完整终点，Hestia 四点、Iris 两臂各四点、canonical 2500/5000 共十四套完整权重和 optimizer 状态。Iris control-640 与 Hestia-320 的完整保留路径见本轮 `retained-endpoints.json`。正式 Gate 导出、数据、模型缓存、Docker/WSL 存储和其他项目未进入删除清单。

云盘删除对象：已废弃的七个 Qwen frozen-feature action-head ZIP、训练链 006/007 两步 smoke ZIP、Don Quixote/Odyssey 各一份重复 ZIP、一个不完整 Hestia 上传分片，以及 Zen 单步 smoke 的 201 个归档分片。重复 ZIP 的成员路径/大小/既有内容身份清单一致。最终 provider 回读确认 213 个目标 ID 全部消失；主备份目录保留 19 项，Zen smoke 目录仅保留 13 个元数据文件。MCP 删除接口为永久删除。一条网络传输错误被保留；重新列目录后，对仍存在的九个实际 ID 重试成功。

本地重复和部分接收文件在删除前与保留完整文件逐字节比较（部分文件验证完整前缀），目标均位于项目授权存储中；删除后 93 个精确路径全部不存在。没有加载模型、执行 ML/训练/仿真、连接 AutoDL、提交或推送；原有用户工作区修改保留。Gate 和 M2 科学结论不变。

本轮证据位于 `.work/checkpoint-cleanup-20261010-001/`：`local-plan.json`、`local-receipt.json`、`retained-endpoints.json`、`cloud-plan.json`、各批 `cloud-*-receipt*.json` 和 `summary.json`。这些是本次维护的新记录，不能替代历史训练验收证据。收据中的 `storage/` 是项目 D 盘存储根的逻辑别名，路径其余部分相对该根。
