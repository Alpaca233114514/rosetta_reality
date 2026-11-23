# 44126恢复示范只读查找结果（2026-09-27）

用户指定44126实例查找未登记ALOHA insertion恢复示范/人工纠正标签。连接实测为无卡模式，没有实验worker；没有启动GPU、模型、仿真、数据下载或修改远端代码/样本。

在Rosetta项目的data、runs、dev、tasks、incoming、transfers、evidence-bundles、control、artifacts、workspaces、vfunfreeze-runtime、artifact_backups、workspace归档目录，以及已解析的closed-axis归档目标中进行只读目录和小型manifest检查。最终目录扫描涉及152,868个文件，未触发45秒/条数截止；另检查60个压缩包的成员目录，全部完成，无候选恢复数据成员。不解压样本，不加载NPY/NPZ数值、图像或Parquet行。

结果：

- 唯一发现的明确数据集manifest为`lerobot/aloha_sim_insertion_human`，revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`，原50episode。
- 四个原始数据Parquet整文件SHA均与20497先前核验的原D相同。不能计作新增S/R或N。
- 所见dataset_view是这个数据源的原train-only视图；数组文件名属于Hestia预测、前缀、KV干预、标定、smoke等诊断证据，没有发现满足恢复数据schema的manifest。
- `runs/m2-smolvla-aloha-geometry-teacher-009/geometry-teacher/33bdc1f4ec9d1315/exact.json`及`...teacher-010/geometry-teacher/ec78415e0a3c5d7a/exact.json`均为`stage=exact,status=failed`，不能作为已验收recovery teacher或成功恢复示范。
- `closed-axis-runtime-archive`为符号链接，目标是历史Iris保存目录，包含旧Zen/vfunfreeze运行资料；不把它当独立数据副本。

结论：在本轮明确检查范围中未找到可直接用于Exp2/3的新增同域恢复示范或人工纠正标签。没有逐一打开全部日志/数组内容或检查整个系统盘，不能把本次查找表述为“任何位置绝对不存在”。当前已有材料仍无法构成合格R_N。

证据：同目录`recovery-search-44126-20260927-001.json`（目录/manifest扫描）、`recovery-search-44126-archives-20260927-001.json`（60归档目录）、`recovery-search-44126-identity-20260927-001.json`（原D四分片SHA）。检查期间设置16:55平台兜底，完成后通过平台请求立即关机；最终状态见截图及关机收据。没有释放实例或删除文件。
