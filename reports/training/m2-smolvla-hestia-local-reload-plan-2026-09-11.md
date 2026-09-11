# Hestia本地模型reload与跨设备控制：登记

`hestia-local-reload-20260911-001`仅在本地固定Linux Docker/XPU执行。远端取回、
清理和关机已完成；用户明确要求继续本地排查，需要新资源再联系。

先只加载原固定1280保存点，不执行中间保存点或新optimizer。逐文件校验取回
的10文件与历史SHA，原配置文件保持不变，仅运行时将设备/本地源路径切换到
XPU，并按历史collector的experiment policy/adaptation配置构建。禁止重新加载
Hub VLM权重；本地模型/依赖/数据缓存、源码、processor与normalization身份固定。

复用历史native factory、processor边界与10步denoiser。45个非hidden frame0、
50槽、真实top相机和两张空相机、相同state/text、四份已保存CPU float32噪声。
输入动作在构建target后移出policy输入。原CUDA collector实际上使用BF16
autocast，即使cfg.use_amp=false；本次对应使用XPU BF16 autocast，不静默换精度。

原始45图SHA必须逐张相同，重建normalized/standard targets与旧值atol1e-6、
rtol0。native trace验证三相机/空图/mask、实际state32维、各图非视觉条件一致、
四噪声实际消费身份；模型参数前后逐tensor哈希不变。

跨设备阈值在执行前固定，区别于原同设备exact reload：active14维normalized
预测atol0.01、standard预测atol0.005，均rtol0.01（逐元素）；四噪声分别对
dev完整/首动作的关节/夹爪、normalized/standard MAE要求相对历史变化<=5%。
先检查episode49零噪声；失败即停，完整结果仍必须全部满足阈值。阈值是此次
跨设备诊断的工程容差，不放宽原科学指标。报告实际max/mean差值，不称逐位一致。

通过本阶段后才能登记中间保存点曲线或进一步同XPU独立进程reload。失败保留
证据、停止quarter采集，不自动改容差、换精度或回到远端GPU。

资源：2CPU、Docker memory及memory+swap各6GiB，XPU allocated<=4GiB，host
RSS<=5GiB，batch1，180个native forward，上限600秒，外层timeout610秒。
一次只加载一个模型，不扩大swap、下载依赖或重训，不打开hidden、不执行Gate。
