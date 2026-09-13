# Iris 本地接入修复模块

模块 `iris-local-repair-20260913-001` 完成本地静态与合成验证。
工作分支 `codex/iris-k-scene-20260913`；用户要求每完成一次模块实验提交一次，
本模块据此本地提交，不推送。此前未提交的其他实验修改继续保留。

## 修复内容

- `image_key_regularization.py`：八层图像K场景方差软约束在原生标量归约后加入；
  lambda=0保留原forward函数。校准值、JSON重复键、布尔/数字类型混淆、层别名、
  SHA及训练身份漂移均拒绝。图像/mask/batch、完整241-token投影、每层一次、
  有限性和梯度合同明确校验。异常清理hook与实例图像方法。
- v2 feature和plan接入：校准相对路径与workspace边界校验，默认不开启。
  当前登记只允许lambda 0或0.01，拒绝叠加其他学习轴或未经验证的compile。
- `iris_runtime.py`：读取权重SHA改为分块；checkpoint在模型/数据构造前核对
  完整文件集合、每文件SHA和对应run/step/batch/episode身份。checkpoint使用
  保存的processor，不以当前统计覆盖其状态。输入排除动作并保留padding mask。
  评估输出create-only；检查完整样本覆盖、有限动作、参数与磁盘权重不变。
  独立reload比较要求两个进程、相同输入与源码权重身份，以及七组数组全部精确。
- loss记录同时保留flow、正则和total；标准`loss`键与实际反向标量一致，避免
  logger合并字典时把总loss重新显示为原生flow loss。

## 验证与边界

命令：`wsl.exe bash scripts/check_iris_preparation.sh`。
执行边界为固定镜像
`sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`，
Linux Docker、network none、2GiB内存与swap同限、2CPU、只读仓库。
Python3.12.3 / torch2.11.0+xpu；没有可用CUDA/XPU，测试使用合成CPU张量。

**91 passed**；Ruff通过，8个Python文件格式检查通过。唯一warning为容器未映射XPU。
覆盖解析梯度、共享token模式、图像切片、异常恢复、错误mask、校准歧义、
checkpoint缺项/篡改/错run以及“首部一致但后部动作变化”的reload反例。
较早18/65测试与风格失败属于中间状态；本记录是最终本地检查，不覆盖历史失败证据。

当前模块没有SSH、模型加载、真实数据、policy forward或optimizer步骤。
已实现的是训练与评估接入边界，**不是完整可启动训练炉**。新鲜基座校准生成器、
双臂阶段配置封存、训练监督器和完整远端阶段链还需下一模块完成；真实CUDA的
batch1 forward、batch4两步smoke、保存processor独立reload仍为`not measured`。
`iris-k-scene-20260913-001`尚未启动，launchable=false；Gate3/4未测，M2未完成。
用户已关机等待本地修复；需要SSH时再通知用户，不自行开机。

先行数值模块已回传的320个CUDA样本精确复现与原float64/BF16一ULP失败是两项
不同证据，原失败保留。已授权空间处理保留232个文件的副本及原路径链接；
该副本在同一实例的系统盘，不能称为异机备份，实例不得释放。
