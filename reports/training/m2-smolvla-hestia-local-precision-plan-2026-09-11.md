# Hestia本地精度与重复性：登记

首个XPU BF16预测与历史CUDA对照未通过冻结容差，失败保留，中间保存点继续
停止。本诊断不修改该阈值，不将原失败改为通过。

仅使用1280权重、train episode49 frame0、四个原噪声。依次固定BF16/FP32，
各噪声重复两次，共16次forward；两次独立进程各重新加载权重，共32次。相同
factory/processor/normalization/原始图像/完整target校验。两模式都保留，不按
对历史预测更近而替换原BF16科学协议。模型参数前后500项逐tensor核验不变。

每个新进程第一份BF16零噪声结果必须与刚保存的失败XPU数组逐值相同，否则
停止并记录重复性问题。每个模式同进程重复要求normalized及standard逐值相同；
两进程最终全部数组也要求逐值相同。对原CUDA的差值只做描述，仍沿用原失败
容差输出布尔值，不作为重新判定原控制的途径。FP32结果不能替代BF16验收。

只加载1张训练图；上下文预检仍读取既有非hidden45场景frame0数值，明确不称
只有一行数据访问。不读取开发图像或hidden。没有optimizer、Gate或远端操作。
若重复性成立，只说明本地数值稳定，不能证明与CUDA等价，也不能将精度差异
归因于模型泛化。后续需单独确定本地保存点比较的可解释边界。

固定离线Linux Docker/XPU，2CPU/6GiB内存及memory+swap，XPU allocated<=4GiB、
host RSS<=5GiB，每进程600秒、外层timeout610秒。失败不自动安装依赖、放宽
容差或改用远端GPU。所有输入与代码在执行前固定SHA。
