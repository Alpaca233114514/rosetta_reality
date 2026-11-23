# D native metadata 索引入口：实际 CPU 优化验收

`d-native-metadata-20260927-009` 对已完成D007自含视图进行了真实CPU核验，不加载模型、不更新optimizer、不重解码原shared source视频，也不覆盖旧view/read proof。

已安装 `datasets.arrow_dataset.Column.__iter__` 在dataset使用custom formatting时保留全部列。原native `hf_transform_to_torch` 属于custom transform；因此仅为了建立episode/frame查找表而遍历原Column也会走图像解码。D facade现在仅对标准metadata列 `select_columns(...).with_format(None)`，实际native `__getitem__` 图像/state/action路径不变。旧benchmark和sealed source不修改；下一benchmark使用新整体身份。

真实结果：

- 完整20,000行metadata的原episode/frame tuple顺序全部精确，查询耗时1.6993秒。
- 原ID选择 `[49,4]` 的filtered视图真实1,000行，返回顺序符合native物理顺序4/49，查询耗时0.1931秒。
- metadata查询期间将 `PIL.Image.Image.load` 设为禁止调用，全部查询通过，未触发图像解码。
- filtered视图6个原生边界读出（frame0、450、499，两episode）按独立原label ledger重算50步动作、末49 padding和state，同时核对原RGB像素SHA，全精确。
- 总24.2063秒，峰值RSS1,030,012,928bytes。注册160秒、6GiB RSS、10MiB输出，metadata阶段rc0，worker/guard owned group均为空。
- 8项真正CPU stdlib tests通过，包含metadata不得触发native完整图像transform的独立假接口测试。

source workspace `20260927T144140Z-66a5bbcdbdcd-1c5e14af3b9d`，composite SHA `1c5e14af3b9d16589c36d6e71472b8321794b131e78cd8b56d246e3a184b3e25`。

新facade源码SHA `ecbd183fd6b9f7253e0c2720823e53bd119ff393be20b7fd7ffa525d55c10c14`；CPU checker `scripts/check_d_native_metadata.py` SHA `c4f81f6eb9a7c1160955d7e91b8b72fa3bacbe3dfd12b190928c64401b114b4b`；tests SHA `3971a1b2710a55bfd4503b6276f2bee42f4a36fa9f1f9b3b7efced7e87c0b711`。

D view manifest仍为 `019e63cf967c144cc283b40c0cb99f5599c6b2581e5cd54df9f7f37aded656fc`，其全20k native proof及D-only stats不变。本报告没有宣称测得旧metadata查询耗时或GPU训练成功。

15个small证据文件逐文件双端SHA一致，接收根 `.work/d-native-metadata-20260927-009-independent-received`，archive SHA `97fdcf0d829faf48e0e507891efa1c8c3a02df8afa2a348d904855f15ad99735`。