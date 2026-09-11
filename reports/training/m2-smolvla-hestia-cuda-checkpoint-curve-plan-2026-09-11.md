# Hestia现成保存点同设备曲线：待授权

状态：仅本地代码、模板和CPU保护检查。尚未获新GPU预算或执行本计划。
前次实例已收到关机命令；本计划不是重训，也不自动重开实例。

## 为什么需要此步骤

本地XPU重复及独立加载稳定，参数未变化，但BF16/FP32均未通过相对原CUDA
记录的冻结容差。两端实际栈为本地PyTorch2.11.0+xpu、原环境2.8.0+cu128，
不能将差异单独归因于显卡或某个算子。保持失败记录，不放宽阈值。

要判断原CUDA训练过程中开发映射是否随着拟合增强而恶化，应使用原计算栈：
RTX4090D、PyTorch2.8.0+cu128、NumPy2.2.6、LeRobot0.6.2，以及已固定原生源码、
processor、normalization、模型/数据revision。若现环境漂移，停止，不安装升级。

## 固定范围与顺序

- 只读取远端已存在的Hestia 1280/320/640/960 native pretrained目录；每个10文件
  逐SHA匹配恢复的quarter清单。optimizer、scheduler、训练状态不加载或修改。
- 固定45个非hidden frame0、原四噪声、50槽、10步native BF16推理；每保存点
  180 forward，完整执行最多720。不筛选开发样本、噪声或保存点。
- 第一个1280控制先检查episode49零噪声，再完成180条件；normalized预测
  全32维、standard全14维相对旧CUDA数组要求逐值相同（atol=rtol=0），其
  dev完整/首动作分组MAE也要求一致。失败即停，后续保存点不启动。
- 320/640/960必须读取已通过控制的结果及数组SHA才能运行。每次独立加载
  相应模型，native输入trace、参数前后SHA、finite/合法action及输出reload检查。
- 本地回收后报告四步的train/dev、关节/夹爪及左右分组、首/完整/末动作变化。
  原Hestia固定1280负结果不改写；中间点较好也不是原实验通过或新的闭环成功。

## 执行与关机

入口为`run_hestia_checkpoint_curve.py`和`diagnose_hestia_cuda_checkpoints.py`。
在新的versioned clean detached workspace，通过`run_autodl.sh shell`设置已登记
平台容器环境，不嵌套Docker，不上传权重或下载依赖。源码用离线Git bundle
传递，不推送GitHub；小型历史C bundle及main1280计划按SHA复制，旧目录保留。

只有收到本次GPU推理、取回结果和关机授权后才启动：

```bash
python scripts/run_hestia_checkpoint_curve.py supervise \
  --template reports/training/m2-smolvla-hestia-cuda-checkpoint-curve-template-2026-09-11.json \
  --execute-authorized --shutdown-authorized
```

长进程以nohup/tmux等价守护启动；独立watchdog绑定父/子PID及启动tick，
失效或到期只停止本次子进程。运行前拒绝其他GPU/项目任务，再执行现场doctor
及15项CPU反例。每个collector验证活跃watchdog、模板、源码、profile、原环境
版本及模型SHA后才加载模型。XPU/CUDA原collector的历史保护不修改。

supervisor总窗口600秒：最多480秒预检/推理，最多120秒取回与关机。每模型
阶段180秒且服从共享期限；allocated<=4GiB、host RSS<=5GiB，逐模型加载。
依据旧180条件采集约一分钟及实测约1.43GB显存规划，不保证四点必能完成；
不足时保留partial和已完成点，未执行点写not measured，不重试或续费。

退出后生成<=64MiB的结果manifest，后台接收端立即读取manifest所列文件并在
本地核验，回写同一manifest SHA的收据后提前关机；没有收据也在硬截止关机。
只允许取回窗口等待，不为本地分析保持实例开启。失败也走相同保存/关机流程。
保持原平台关机包装器SHA、Trash和其他任务保护；不释放实例。

## 准备验证与限制

本地固定离线CPU容器已检查许可缺失/扩大、过期、错误步数、路径逃逸、旧PID
不误杀和数值单元素越界等15项反例，Ruff/格式及CLI检查通过。它们不代表真实
CUDA控制通过，也不代表完整watchdog/传输握手在远端实测通过。运行时再次核验。
原hidden、Gate3/4和M2状态保持不变。
