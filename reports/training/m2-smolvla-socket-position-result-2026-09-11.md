# 同一冻结 K/V 能读出开发场景的插槽位置

`socket-position-readout-20260911-001` 已完成并通过登记的辅助位置正对照。
相同早层 K/V 在首动作读出失败的同时，保留了可跨场景读出的蓝色插槽位置。
因此“该表示完全没有可读插槽位置信息”不符合这次证据；动作映射失败的唯一
原因仍未识别，也没有证明原生动作专家已经充分利用位置。

## 数值证据

所有数值为原 640×480 图像上的 pixel MAE。仅从 40 train 选择 alpha=0.001，
五个外折也均选 0.001（网格下界，未继续扩网格）。位置标签不是动作标签，
不能把其数值大小或选出的 alpha 与前两次动作读出直接比较。

| 集合 | 坐标 | K/V 读出 | train 均值常数 | train 中位数常数 |
|---|---|---:|---:|---:|
| train 嵌套留出 | x | 4.7350 | 10.5652 | 10.7590 |
| train 嵌套留出 | y | 3.9917 | 19.2094 | 18.7178 |
| dev5 | x | 4.3848 | 12.4297 | 12.9521 |
| dev5 | y | 5.7143 | 12.9236 | 12.9129 |

嵌套行的常数只在每个外折的训练部分拟合。两个坐标在嵌套留出和 dev5
均低于两个常数基线的 50%，全部通过登记条件。dev5 的全部非自身错配图
平均 MAE 为 x=14.5459、y=19.0002；正确图优势分别为 10.1611、13.2858 pixel。
dev5 场景相关系数 x=0.90380、y=0.93905。全逐场景结果保留在同名 JSON。

## 标签和身份核验

复用上次 `contextual-kv-depth-20260911-001` 的早层 1、2×2、2560 维特征，
没有新模型 forward。40 train 与 5 dev 的 frame-0 图像 SHA、state、action、
episode/frame 顺序全部与原 K/V 采集一致；缓存 manifest/checksum 验证通过。
固定蓝色规则抽取像素质心，未用动作或开发误差调整阈值；主/严格规则的
最大每轴差为 1.06785 pixel，低于预定 2 pixel 上限。

已实际查看三张检查页中的全部 45 个首帧，黄色 bbox 均对应蓝色插槽，白色
质心均落在插槽区域，没有误选背景或红色 peg。原图、bbox 和质心页保留在：

- `runs/socket-position-readout-20260911-001/position-check-1.png`
- `runs/socket-position-readout-20260911-001/position-check-2.png`
- `runs/socket-position-readout-20260911-001/position-check-3.png`

这是二维可见颜色区域的代理位置，没有三维标定、姿态、遮挡泛化或 peg 位置信息
验收。官方资产用于预先确定蓝色语义，实际提取的正确性以这 45 图为依据。
依据：[ACT insertion asset](https://github.com/tonyzhaozh/act/blob/main/assets/bimanual_viperx_insertion.xml)。

## 运行与结论限制

35 项 synthetic/regression checks 和 1 项真实缓存测试通过；Ruff、格式及
check_env 通过。原镜像 digest、CPU 2 核、2 GiB 内存及 memory+swap、网络关闭。
采集和分析用时 16.402 秒；`Could not load libtorchcodec` 警告后由已有 PyAV
后端完成解码，45 图哈希逐一相同。没有安装依赖或改变运行环境。
输出数组 exact reload 通过；机器结果中的视觉检查 `pending` 为采集时状态，
本报告和同名 JSON 追加记录上述已完成检查，不改写原始结果。

dev5 已参与开发，不是全新独立测试集。上游 Zen 的训练曾包含 40 train 场景，
嵌套折只隔离本轮线性读出。该结论不能外推为完整空间表示、策略视觉泛化、
动作正确性或 M2 成功。没有 SSH、optimizer、模型 forward、仿真或新 Gate。

下一诊断是固定位置与示范动作的关系：先区分起始动作与后续目标导向动作，
不因为位置可读就归因于某个训练模块，也不把相关性弱直接解释成错误标签。
