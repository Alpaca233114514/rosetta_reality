# Hestia保存点曲线002：修复部署入口

用户重新提供SSH并授权补救。只修复001缺失的normalization入口，仍对同一组已有
1280/320/640/960保存点执行原45帧、4固定噪声条件。完全保留原CUDA逐值相同控制、
processor/参数/模型/数据身份、资源上限、失败即停、最多480秒计算及600秒关机窗口。
不训练、不下载、不选择保存点，不自动重跑。001失败证据与workspace全部保留。

本次新输出为`runs/hestia-cuda-checkpoint-curve-20260911-002`。
`prepare_hestia_checkpoint_workspace.py`先校验durable normalization report与view
manifest的既定SHA，再在新workspace创建report文件链接及view目录链接。任何已有
入口、越界或SHA差异均停止；不修改durable内容。随后实际执行native
`_validate_normalization`，覆盖view父目录身份、全部文件SHA、训练split与统计合同，
生成`normalization-check.json`。此步骤在独立看门狗启动之后、权重加载之前运行。

本地回归使用真实既定plan/report/view：新workspace缺失入口时必须失败，绑定后
完整native检查必须通过；重复绑定与SHA漂移必须拒绝。另运行既有20项保护检查。
现场先doctor、再完整normalization预检与既有CPU检查，全部通过才启动1280控制。
结果后台回传与关机握手沿用001已实测路径，新增normalization预检结果进入SHA清单。
新模板绑定本次源码。真正推理结果以002现场证据为准。
