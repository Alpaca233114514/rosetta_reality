# Hestia V 分量诊断入口失败与后续修正

`hestia-value-component-20260912-001` 的第一个原生 forward 在 V hook 中触发 `ValueError: Native V prefix shape changed`，没有完成任何条件或完整 policy forward，0 optimizer。81.208 秒后退出。8 文件全部回传、逐 SHA 校验，匹配收据已送回，关机请求也已单独恢复；平台随后实测已关机。原运行与代码不重写。

诊断代码错误地要求完整 V 输入为 `(1,64,320)`。历史 prefix observer 实际仅保存 `value[:, :64]` 的真实相机切片，完整原生前缀还包括空相机、语言和状态位置。这是新增诊断入口错误，不是 Hestia 视觉泛化的根因证据。

固定 SHA 上游源码和 checkpoint 配置核对后，完整布局确定为 `3×64 +48 +1 =241`。状态投影的 weight/bias 在两个 checkpoint 间变化，不能据冻结 VLM 权重要求整段 prefix 跨 checkpoint 恒定。语言/状态位置含上下文，也不能简单当作纯文字/纯本体状态特征。

新身份 `hestia-value-route-20260912-001` 改为完整 241-token 路由干预，各部分精确复制实际 CUDA donor 输出，前四项为两个原生端点和两个完整 V-only 对照。150 项本地检查通过，包括 64-token 截断拒绝、完整布局/空相机 mask、状态漂移拒绝、精确分段复制和阳性对照失败阻断。新运行前四对照的全部 720 次输出已经精确匹配历史数组；其余六项仍按登记顺序执行，不能提前报告完成。

失败原始证据见 `runs/hestia-value-component-recovered-20260912-001/verified/`，机器摘要和闭合见同日期 `value-component-failure.json`、`value-component-shutdown-recovery.json`。本地不完整首包因模板键类型错误而另名保留；修正后有效包 92,160 字节才被上传，未在远端使用不完整包。
