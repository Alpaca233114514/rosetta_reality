# Hestia本地1280控制：首个跨设备预测未通过容差

已在本地固定离线XPU容器执行，没有再次连接远端或启动optimizer。
Ruff/格式、22项相关测试以及1项真实缓存检查通过。40文件取回身份已另行通过，
本阶段加载1280步的10个native文件时再次检查其SHA。

程序完成45张frame0原始图像与历史C摘要核对、完整target重建比较（atol1e-6），
随后在episode49、零噪声的第1次native BF16 forward触发停止：active14维
normalized输出相对原CUDA记录最大绝对差0.037109375、平均绝对差
0.006118926916803632；预登记atol0.01/rtol0.01的逐元素判定失败。

失败值及原记录保留在
`runs/hestia-local-reload-20260911-001/first-control-failure.npz`，不是新验收通过。
其余179次forward及320/640/960采集均未执行；没有放宽容差、替换精度或
自动改走远端GPU。完整native非视觉trace跨图一致性及参数前后对照位于后续
代码段，本次未到达，不能称这些最终检查已通过。

原配置注明use_amp=false，但历史collector明确包裹CUDA BF16 autocast；本次
使用对应的XPU BF16，不能将该差异解释为训练变化。当前只证明跨设备数值
控制失败；尚未区分后端/精度差异、重复性或其他运行路径差异。

下一项本地诊断将固定同一权重、同一episode49及四个既有噪声，对BF16/FP32
做重复及独立进程对照。不会用该诊断改写本次失败或原Hestia结果，也不据此
立即比较保存点。

启动日志保留常见的saved CUDA配置转入明确请求XPU的警告，以及已有PyAV
解码回退；没有安装或升级依赖。证据位于`runs/hestia-local-reload-preflight-20260911-001/`，
同名JSON固定所有结果SHA。
