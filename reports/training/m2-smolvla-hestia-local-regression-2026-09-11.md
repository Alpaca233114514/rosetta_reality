# Hestia 主链本地合成回归通过，远端验收仍待执行

在已登记的本地Linux Docker中，当前主链对应的197项合成/回归检查通过，
failures/errors/skipped均为0，包括此前未执行的13项主链反例。
5120个合成样本的原生sampler顺序与独立构造完全一致，40个train episode
各128次，顺序SHA与历史登记值相同。

测试命令来自 `scripts/check_hestia_collector_cpu.py` 的11个TESTS模块，
但没有调用它的AutoDL专用完整验收入口，也没有设置假的AutoDL环境身份。
随后调用 `scripts/inspect_hestia_schedule.py`，其中原生sampler源码SHA检查通过。
测试总耗时约19.87秒；本地CPU容器提示XPU设备数为零，检查仍通过。

初始Ruff通过，格式检查发现三文件待整理：`scripts/run_hestia_fit.py`、
`src/rosetta_reality/vla/visual_fit_job.py` 和 `tests/test_smolvla_visual_fit_job.py`。
格式化后逐文件AST完全一致，未修改模型路径、门禁、资源或实验行为。
最终12文件Ruff/格式检查通过。197项测试是在格式化前执行；AST不变已验证，
没有把未执行的第二轮测试计入。初始格式失败日志保留，未覆盖历史结果。

运行镜像digest为
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`，
PyTorch2.11.0+xpu、Python3.12.3，2CPU、主存/memory+swap各2GiB、离线。
check_env通过；没有加载真实权重/样本。合成测试内部包含小张量AdamW与
scheduler回归，不是原生SmolVLA optimizer smoke或训练。

证据保存在 `runs/hestia-local-regression-20260911-001/`，同名JSON记录前后
源码SHA、日志/原始XML SHA及scope。原始XML含容器主机名，只保留于ignored runs。

本结果补齐本地代码检查，**不替代当前AutoDL完整验收**：实际B文件/恢复状态、
saved-config roundtrip、当前CUDA环境/缓存身份、磁盘/活跃任务和smoke复用前提
仍须在获准实例上重新核验；本地Torch也与历史CUDA运行不同。
无SSH、C训练、B/C新采集、模型reload或Gate。Hestia未启动，M2未完成。
