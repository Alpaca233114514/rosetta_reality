# Hestia 预检修订 002

前次 `runs/hestia-gpu-preflight-001` 在源码 `264e06d1355131e243acf8934ede21ae31c61150` 的 Ruff 阶段退出 1，没有执行 pytest、模型加载、GPU forward 或 optimizer。原始失败与身份文件保存在 `reports/training/hestia-preflight-20260910/001-*`。原始报错为 E402 两处、E501 一处、I001 两处。格式化成功并不意味着 Ruff 检查通过；stdin 格式化所用旧工作区还不包含新模块，造成 first-party import 分类不同。

修改仅涉及导入位置、导入分组、长字符串拆分，以及把完整代码验收提前到 GPU job 创建之前。不改模型、数据、学习轴、optimizer、scheduler、验收门槛或旧失败证据。前次 watchdog 在 11:52:51 UTC 开始，worker 当秒停止；11:55 左右的后续 SSH 返回 `Connection closed`。未回收关机入口执行记录，不能把 SSH 关闭独立认定为平台计费停止。

用户随后重新提供 SSH，继续本模块。执行新 clean commit 工作区，依次：

1. 在已登记 AutoDL 环境执行 `python scripts/run_hestia_preflight.py verify-code`：Ruff、原登记七组 CPU 回归、无 skipped/failed、原生 trainer/sampler/scheduler/modeling 源码 SHA 校验。每个子命令最多 180 秒；不加载 SmolVLA 或真实数据，不执行 GPU 模型。输出 `runs/hestia-code-validation-002/`。
2. 仅前一步通过才执行 `prepare`，并核验受测源码全量 SHA 和依赖版本仍相同。生成独立 `runs/hestia-gpu-preflight-002/`；GPU 实测身份由新 registration 记录。
3. 启动 `supervise`。继承原计划 1200 秒 GPU 模块预算与完成/失败后 120 秒回收窗口，以及原平台关机、禁止释放、禁止重试规则。worker 核验先前 CPU 证据，不重复运行已通过回归，随后执行 doctor/data/benchmark、batch 1/4 forward、真实输入合同、两步 observed smoke、checkpoint 审计、两个独立 full-chunk reload。

各 GPU 子阶段名称后缀使用 `-002`，不覆盖 001。Hestia 主候选炉名保持 `m2-smolvla450m-visual-hestia-fit40-001`；当前仍不运行 1280-step 主训练。完整候选协议与解释见 `reports/training/m2-smolvla-hestia-gpu-preflight-plan-2026-09-10.md`。

如果 CPU 代码验收失败，只保留日志、修复代码并重新登记源码；不启动 GPU 模型。GPU 前置失败后停止本模块，保留结果并按授权关机。M2 和视觉泛化尚未修复。
