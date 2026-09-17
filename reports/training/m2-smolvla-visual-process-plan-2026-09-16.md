# Step2500→5000固定配对过程诊断

## 预先固定的研究问题

主对象仍为step5000。补采既存step2500，仅用于描述同一训练过程中的拟合和开发
表现变化，不重选checkpoint，不新增训练或Gate，不打开隐藏集。
前轮观察过step5000，故具体125/250夹爪和frame0关节问题属于有先验探索的开发诊断。

| 固定项 | 合同 |
| --- | --- |
| 数据/采样 | 原train40/dev5，每集0/125/250/375；180输入 |
| 采集实现 | 与step5000相同的`visual_research_collect.py`，SHA `ac3586de47c0c862a98a3c7578b75065c00094731dd860244b22d8651f945e16` |
| 噪声 | 零及20260905/06/07；完整noise数组必须与step5000精确相同 |
| 非视觉与标签 | state、语言、targets/raw/normalized targets、mask、donor、图像身份精确一致 |
| 唯一主变化 | checkpoint权重2500→5000；相同保存的processor状态 |
| 配对 | frame0全部非自身，后续相同循环donor；不赋予反事实动作标签 |
| 子群 | 固定step2500已见83/未见77个训练输入，开发20输入；逐时点单列 |
| 基线 | 同时间/slot的train-only均值及中位数，训练查询排除自身episode |
| 汇总 | 逐episode、维度组、噪声、首动作/完整chunk；episode等权及留一范围 |

主要结果保存正确/错图MAE/MSE及视觉收益的两端值、5000−2500差值、完整逐episode
结果；同时保留预测/目标方差、协方差和均值偏差。固定子群在配对指标算完后切片，
不能改变donor池。逐维结果可由完整NPZ复算。

## 身份与资源

- 2500权重SHA：`a3494600183d0a24c71ab7db0976afbc57567ac714ac678fd572c607e06a2f0c`。
- 5000权重SHA：`d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef`。
- 2500完整文件清单来自已回传`canonical-fullframes-20260914-001/checkpoint.json`。
  其清单不包含5000追加的image-preprocessing sidecar；采集显式使用同一个
  canonical RGB函数和processor。实际采集前仍须通过文件集合与SHA检查。
- 科学登记：`runs/visual-research-process-preparation-20260916-001/scientific-registration.json`。
- 运行配置：`configs/vla/visual_research_gpu_20260916_002.json`，SHA
  `030f7b8ad36fe61e96a4d0d973c7b5dab9d510022692ad4cb35810e94ef36958`。
- 先前GPU窗口保守计20分钟，本次最多30分钟，合计不超过60分钟；因中途缺卡，
  本轮未延长期限。计算最晚北京时间10:57停止，11:02硬关机，预留300秒收尾。
- 1268次forward、0optimizer；首次计时估算剩余采集耗时，加25%余量仍须能放入
  计算窗口。GPU分配8GiB、保留10GiB、RSS8GiB、输出256MiB上限。
- 身份、非finite、self-copy、资源或预算失败立即停止并保留失败记录。

只启动原实例。启动最初因空闲GPU为0被平台拒绝；准备期间卡恢复，再次请求
启动成功。没有新租用或克隆实例，也未改变GPU型号。

## 验证与解释

`scripts/compare_visual_research_checkpoints.py`独立用NumPy复算两个端点，再对账
各自生产分析。其合成CPU回归3项通过：已知MAE/MSE增量与固定子群、共享noise
不一致拒绝、episode聚合避免伪重复；Ruff通过。合成测试不是真实2500结果。

先回传逐文件校验并保护式关机，再在本地CPU复算。SSH断开不证明关机，需平台
新状态。即使2500离线指标较优，也不改变主checkpoint，不称为模型修复。
过程差值不能确定增加场景的因果效果，也不能单凭G4 0/5判定训练代码有错。
