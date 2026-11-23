# ALOHA train隔离准入登记（2026-09-27）

身份：`aloha-train-isolation-20260927-004`。本轮用户要求解决阻塞后继续实验，并明确批准
“允许隔离预处理，仅导出 train”。该批准只改变Q阶段原混合row group不读的边界；
不授权读取hidden用于分析/训练、不伪造标签、不改变Gate阈值。

唯一工程变化：在隔离子进程内部允许底层Parquet解码混合页，scanner只返回train40，
写出之前再次核验episode白名单。独立进程仅读取派生文件，检查所有episode/frame、
时间、state/action有限性、固定维度、长度与有效监督slots。旧入口保留。

来源仍为两份已核验下载：human revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`、
scripted revision `8ab660912970111cbb26738b11458e6fc4a4aed1`。源receipt SHA、
train/dev/hidden清单、Action Contract SHA全部在配套JSON固定；human期望40×500=20,000帧，
scripted期望40×400=16,000帧。编号排除对scripted是保守规则，不据此证明scene隔离。
human属于D，不能算新增N。next.done只作终止字段，不当成功证据。

输出为远端全新`runs/aloha-train-isolation-20260927-004`。源码经既有verified-source-delta
入口形成新content-addressed workspace，上传后逐文件SHA核验再启动；禁止覆盖旧run。
真实样本不回传桌面；只回传合成测试日志、源/导出身份清单及逐train-episode审计统计。

顺序：旧标准库回归（45秒上限）→真实合成Parquet隔离测试（45秒）→真实导出及
独立核验（180秒）。主任务总工作截止300秒，独立guard关机截止420秒，平台另设
约10分钟截止并核验。正常完成预留最多240秒回传，收到本地SHA核验标记立即关机。
前轮源hash/footer审计不足1秒，这次仅36,000行、7个小Parquet，预算主要用于测试、
启动及回传，不按正式训练小时数占用GPU。

隔离进程单文件及总导出≤32MiB、批次128、关闭并行scanner、RSS检测≤1GiB、
CPU时间≤180秒、禁core dump；远端至少1GiB磁盘余量。使用已安装的PyArrow环境，
不安装依赖、不下载模型/数据、不嵌套Docker。profile记录身份参考，未跑CUDA doctor
不得声明模型runtime准入。身份漂移、白名单泄漏、帧/时间/维度不符、nonfinite、
资源/截止超限或测试失败立即保留失败证据并停止，不自动retry/resume。

此阶段验收只证明隔离与结构。S/R标签、视频对齐、scene/parent保护及matched N仍须证据。
原base Gate3的9个关节越限计数和1个跨臂夹爪接触是原报告中的实际失败，不能改为passed；
在没有已证实实现缺陷的情况下，base Gate4保持not_measured，其他实验不借此跳过Gate。
