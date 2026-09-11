# C开发失败是否位于训练几何覆盖之外

登记 `hestia-geometric-support-20260911-001`。仅在本地离线容器使用缓存，
不重新开机或连接远端。检验“开发失败主要是socket图像位置超出训练覆盖”
这一有限解释，同时检查固定几何近邻是否比C更接近目标动作。

先重新读取固定45个非hidden frame-0图像，验证本地KV采集的uint8 SHA及
Hestia C记录的原始输入tensor SHA两种摘要；frame/episode/camera/state/action
与缓存合同必须匹配。复用已人工检查的固定蓝色mask，质心、bbox、面积与
前一轮位置提取记录逐值一致。差异即停，不更改mask或样本。

只用train40建立覆盖描述：

- 二维质心的轴范围与凸包，边界计为覆盖内；不用dev扩展凸包。
- 二维质心、五维图像几何代理(cx,cy,width,height,mask面积)分别用train均值/
  标准差缩放；保存train留一近邻距离及dev到train最近邻距离。
  train标准差为0的列使用缩放1并单独标记，不用dev决定缩放。
- 参考距离固定为train留一距离95%分位数，不用dev选阈值。凸包或距离只
  描述这些观测代理，不代表完整3D姿态、遮挡、动力学或所有图像特征均被覆盖。

固定1-NN动作读出：两种距离各使用唯一最近train图像的完整目标chunk；同距
按登记episode顺序取第一个，不搜索k或距离权重。报告train留一、dev5，
关节/夹爪以及左右分组、full/first/early/middle/late/last；对照两种train
常数和C四噪声。近邻只使用图像几何选donor，dev标签不参与选择。

无新的成功门槛；按全部dev场景保留位置、donor、距离、预测与误差。若某些
场景在上述覆盖内却失败，则不能用二维位置越界充分解释所有失败；反之也
不能仅凭越界证明因果。1-NN不是正式policy，也不作为闭环修复。

镜像digest沿用已有本地运行，2CPU、主存/memory+swap各2GiB、离线，180秒。
先Ruff/格式、凸包边界与近邻隔离反例，再执行一次。源码/输入SHA冻结、
create-only证据，权重/模型不加载，optimizer/Gate/SSH均为0，hidden封存。
