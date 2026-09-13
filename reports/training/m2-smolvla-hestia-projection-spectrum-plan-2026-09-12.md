# Hestia 原生 K/V 投影谱检查预登记

在等待当前SSH信息期间，继续本地排除一个具体机制：640→1280是否伴随
早/晚 cross-attention K/V 投影秩坍塌或显著条件数恶化。名称
`hestia-projection-spectrum-20260912-001` 已查 reports/configs/scripts/tests 无重名。

只从已取回640、1280 checkpoint读取第1/15层k_proj/v_proj的weight，共8个
320×320矩阵。先流式核验完整checkpoint SHA，再以safetensors读取这8个指定
tensor；不构造模型、不加载其他权重张量、不读取样本、不推理或训练。
header已确认这些投影是方阵。本轮不访问SSH、hidden或仿真。

固定 float64 SVD，报告完整singular values、数值rank（阈值为
max(shape)×float64 eps×s_max）、s_min/s_max、condition number、stable rank，
以及小于1%最大奇异值的维度数。保留零/秩亏结果，不把无穷condition编成0。
配对报告Frobenius相对权重变化、条件数比值；条件数≥10倍仅作为预登记
严重恶化标记，不是闭环门槛。合成检查覆盖奇异/恒等/尺度/旋转不变性。

若rank完整，能排除保存权重在实数线性映射意义上的确切维度丢失；不能
据此断言BF16无误差、attention路由正确或信息被有效使用。条件数和stable
rank描述矩阵，不是实际激活分布。结果不选择checkpoint或更改任何根因
验收条件，不能把此检查代替待授权原生CUDA激活。

同名JSON绑定源码、测试、checkpoint原SHA和输出。固定既有离线Linux镜像，
WSL Bash启动Docker，2CPU/2GiB，总上限180秒；只有新的输出/前置目录可写。
输入/源码漂移、shape/finite异常、资源超限即停，保留证据。独立读回奇异谱
并用trace/Frobenius恒等式检验；不向Git保存权重。
