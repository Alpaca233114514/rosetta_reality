# 只用训练残差的共同轨迹偏差校正诊断

登记 `chunk-bias-transfer-20260911-001`。上一轮发现原生输出有明显共同均值
偏差；本轮检验其是否能由 train 学到并迁移到 dev，而非使用 dev 均值作 oracle。

复用完整 chunk 的 projected_targets 与 native_standard，不加载原始数据、
图像或模型。固定 40 train / 5 dev，50步，14维；目标、噪声和artifact不变。
唯一处理为每个chunk槽位/维度的共同加性偏差：
`b[t,d] = mean_train(target - native)`，`candidate = native + b`。
无场景相关系数、幅度缩放、偏移网格或 dev 调参。偏差只适用于这次 frame-0
条件，不作为通用策略写入部署路径。

控制与处理均执行同一既有Action Contract bounds投影后比较，并分别报告
发生投影的元素数。处理投影前数组也保存；不能把裁剪后的合法性称为真实控制
安全验证。原生控制若本就合法，此步骤不改变它。校正本身不创造新场景信息；
不经clipping时场景预测差分/协方差不变。

报告full/first/early/middle/late/last（与上一轮相同窗口），关节和夹爪分组。
额外做校正leave-one-out：每个train行的偏差只由另外39行算；常数同样使用
另外39行。这只隔离校正，不能抹去上游策略曾训练过这些场景的事实。

两主组均满足下列条件，才支持这个有限的偏差迁移诊断：
1. full dev MAE较投影后的原生控制降低至少20%；
2. full dev MAE低于train mean和median，epsilon=1e-8；
3. full dev正确图优于全部非自身图平均MAE，epsilon=1e-8；
4. 校正LOO的MAE低于对应两种常数，epsilon=1e-8；
5. first-action dev MAE不比原生控制更差（容差1e-8）。

整体失败就保留失败，不改变门槛或拆组挑结果。通过也仅是零噪声frame-0的
离线诊断；dev已参与开发，不能认定视觉泛化或闭环已修复。Hestia、Gate和
模型权重不变。新观察不替代前一轮完整ridge失败或早先0.8阈值失败。

已有Linux Docker / WSL Bash，固定镜像，CPU2核，内存和memory+swap各2GiB，
网络关闭、180秒上限。check_env、Ruff、合成隔离检查和输入SHA通过后运行一次。
create-only保存偏差、校正/原生完整数组、LOO、指标和array exact reload。
无下载、SSH、策略optimizer、模型forward或仿真；M2未完成。
