# 本轮训练空间的删除候选清单

状态：只读盘点，未删除。用户拒绝扩容，要求先给具体清单。

下表路径均相对于服务器项目根。候选是本轮已登记 ALOHA D/S/R 三臂训练不使用的外部数据缓存，不是已经确认存在完整本地副本的重复文件。后续再次使用这些数据需要从原来源恢复；本次未验证其完整本地备份或未来下载可用性。

| 编号 | 精确相对目录 | 普通文件数 | 实际文件字节 | 文件占用字节 |
|---|---|---:|---:|---:|
| 1 | `datasets_external/targeted-20260927-001/REBOOT26/USB-A_recovery_install/cfa3498a3982eb24554e88e77140247871eee3eb/data` | 60 | 34473940173 | 34474201088 |
| 2 | `datasets_external/targeted-20260927-001/REBOOT26/USB-A_recovery_install/cfa3498a3982eb24554e88e77140247871eee3eb/videos` | 16 | 2809152742 | 2809184256 |
| 3 | `datasets_external/targeted-20260927-001/REBOOT26/rj45_recovery_install/f6acd5b69394ffd4d2d8bd30079306c9e2061bbe/data` | 24 | 13986383026 | 13986533376 |
| 4 | `datasets_external/targeted-20260927-001/REBOOT26/rj45_recovery_install/f6acd5b69394ffd4d2d8bd30079306c9e2061bbe/videos` | 17 | 2913762952 | 2913812480 |
| 5 | `datasets_external/targeted-20260927-001/open_x_embodiment/austin_sirius_dataset_converted_externally_to_rlds/0.1.0` | 66 | 7031694109 | 7031824384 |
| 6 | `datasets_external/targeted-20260927-001/amandlek/mimicgen_datasets/33016f8a62c02334f929f2913af8fdd2a8a129e1/core` | 3 | 4851394230 | 4851408896 |

本次实际扫描六个目录均没有符号链接成员、多个硬链接的文件或可见的打开文件句柄。删除前必须重新检查目标解析路径、文件清单、依赖和句柄，不能将本次快照视为永久保证。

建议仅选择编号1，释放约32.11 GiB，已经覆盖三臂十二个完整checkpoint约19.37 GB的预算及余量，不需要同时删除其余候选。原D train40、Q1/Q2/bank、模型、环境、全部实验失败证据和本轮checkpoint不在候选范围。

即使获授权删除编号1，仍保留同revision目录的meta、videos、下载manifest及现有本地下载清单。生成单独删除收据说明该缓存已不完整，不改写旧下载manifest冒充完整缓存。USB下载manifest本次SHA为 `6d6525a87caf72d4bd1155b778dabfaf731aaf46b28f2f96e01070206b053938`。

更可逆的替代是保留缓存并扩容，或先迁移到其他可靠存储；前者已被用户拒绝，后者需要传输时间。不能仅凭“备份可以先不管”推定获得数据删除授权。
