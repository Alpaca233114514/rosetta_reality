# Hestia 同阶段命令末端位姿检查（2026-09-12）

登记名 `hestia-command-pose-20260912-001`；是已授权本地诊断，不是新训练炉。
在读取新统计前固定本计划、运行身份、代码和输入 SHA；不改变历史证据。

假设：阶段 oracle 后左腕标签失配可能仅是关节表达差异；若同样输入邻域的
末端位置和朝向仍不稳定，则单纯将标签转为末端位姿不足以消除该问题。
唯一比较轴是同一批绝对关节命令与其正运动学末端位姿，不改变 donor、样本、
窗口、checkpoint 或动作。clock 和 event oracle 沿用已登记的各 30 槽数据。

输入只使用已绑定的 train40/dev5 数组与元数据。固定二维物体坐标 3-NN，
train 留一，dev donor 仅 train；核对既有 donor 与关节均值逐值相同。
不解码视频、不读新原始数据或 hidden，不拟合参数，也不使用策略权重。

采用固定容器原生 Gym-ALOHA insertion MJCF；记录整个 assets 树与软件版本。
逐维由 Action Contract 名称映射 hinge qpos，核对单位、范围与元数据；
调用 `mj_kinematics` 读取左右 `gripper_link` 在世界坐标的位置和旋转。
这是将命令视作已到达关节位置计算的 FK，**不是实际 state、动态可达性、
手指接触点、抓取物位姿或 rollout**。掌部 frame 的祖先必须只有各自六个
arm joints，因此无需把夹爪编码当成腕部关节。其他 qpos 固定原生默认值。
不裁剪、IK、接触求解、`mj_step` 或渲染；仅合成 sanity 对照一次 `mj_forward`。

对照保留三组：FK(3-NN joint mean)、FK(train joint mean/median)，以及
先逐标签 FK 再平均位置与 SO(3) chordal mean 的 neighbor/constant。
后者用于区分非线性关节平均带来的影响；不平均 Euler 角，不合并米和弧度。
报告位置 L2 米、旋转 geodesic 弧度，scene 等权均值及逐 scene 值；全部
组别共同保留，不挑 dev 最优表达作为新模型。输入最近十对 train 场景仅按
几何距离排序，报告同一时槽的位姿差异。不同阶段坐标不直接比较绝对误差
来声称策略改善；event oracle 仍使用 query 真值阶段，不能部署。

前置：固定 image 身份、runtime/asset SHA；Ruff、合成 SO(3)/汇总反例、
`check_env.py`；真实原生模型用 0.01 rad 左腕扰动验证角度、右臂独立、
`mj_kinematics` 与 `mj_forward` 一致，物理时间保持零。
停止：身份/shape/finite/关节范围失败、退化旋转均值、超时/内存越界。
离线 Linux Docker，2 CPU / 2 GiB，核心最多 120 秒、外层 150 秒；输出
全新目录且独立重载逐值相同。脚本和计划只在执行前修订；失败另留证据。

无 SSH、下载、正式训练、推送或新 Gate 3/4。判断边界：两类末端误差仍差于
对应常数时，不把关节多解当作已证实原因，不据此登记任务空间损失训练。
若位置较好但朝向仍差，则下一步区分朝向信号、演示策略和相位表示，不能
丢弃朝向合同或把它的常数基线称为解决泛化。
