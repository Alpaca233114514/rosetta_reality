# Hestia 同环境原生前缀读出预登记

采集已完成且完整normalized/standard180输出与原C逐值相同；45组前后投影
激活已回传，四噪声/10 denoise步内均恒定。现在只读本地保存数组。
名称`hestia-prefix-readout-20260912-001`已搜索无重名。

沿用prefix-path预登记的四处表示：layer1/15的K/V输入、投影后输出；
每处固定mean和2×2 ordered pooling，八臂全部报告，不按dev选层或pooling。
只合并同层K/V，实摄像头8×8 token保持顺序。所有输入必须对应原C的45行。

目标固定为既有双物体像素位置、左右首次0.5开合帧、原生0–49动作以及
上一轮event oracle的30槽动作。每个物理组独立选择alpha，网格
`[0.001,0.1,1,10,100,1000,10000]`。train外层逐场景留一；内层五折、
seed20260912+query_row，dev只用train40，内层seed20260912。
标准化与选择均在允许donor内；不使用hidden或按dev改变方法。
为避免相同核矩阵重复求解，独立计算ridge线性权重并复用到所有目标组，
须以旧ridge_predict数值对照和标签隔离反例先验证。并行目标不混合单位
选择alpha。动作先保存raw再按原合同限幅并计数，常数mean/median同donor。

报告train外层/开发每组MAE、常数比、全部dev场景、full/first动作及末端
位置/朝向；位置另报告每个像素轴，避免只报合计。核验原始C数组和回传
array逐值相同；原raw image hash、非视觉恒定和权重不变证据来自本次同CUDA采集。

这只能定位可读性与原生输出的差别，不能由一个探针失败证明信息不存在，
也不能凭pre/post差异宣布因果。已知检查的320方阵float64满秩；pooling
与正则化不等价于实际attention。每一层/组独立报告，不汇总成新Gate。
已有dev参与诊断，结果不是独立验证。没有模型forward、训练、原始样本或SSH。

固定离线Linux Docker，2CPU/2GiB，核心300秒，总600秒，SHA/finite/shape/
资源异常立即停止。输出create-only；原始权重、激活、历史结果不覆盖。
