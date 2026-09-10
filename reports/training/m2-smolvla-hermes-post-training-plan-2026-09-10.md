# Hermes：coverage40 训练后审计与评估续接

本模块只复用已完成的 B `visual-coverage40-main256-003` 及历史 A checkpoint，新增 optimizer steps 为零。用户授权按实际任务安排计算、提交和推送；后续炉名使用希腊神话人物。本模块是同一覆盖实验的评估续接，不是新训练炉。

原 job 003 的 256 步训练与冻结范围审计完成，随后采样审计停止。保留原 closure。实际 sampler 记录 1028 个索引，其中前 1024 个与预登记顺序逐项相同。已安装 Accelerate `DataLoaderShard` 在 yield 当前 batch 前读取下一批；训练器循环只消费 256 批。本次须先用该实际类、batch 4、workers 0 的无模型合成数据重现 1024 consumed / 1028 observed，并验证完整预取顺序、256 次真实 LR、最终 scheduler、配置和 checkpoint。

只在上述审计通过后封存新的执行合同。评估代码、噪声、非视觉条件、45 个 frame-0 样本、所有非自身错配平均误差、关节和夹爪分组、均值下限、至少 3/4 条件、跨臂增益、物理误差不回退以及完整数组 independent reload 规则全部不变。hidden 不加载；dev5 仍属开发验证，M2 未完成。

资源上限沿用 allocated 8 GiB / reserved 10 GiB / RSS 10 GiB。新监督器独立计时最多 30 分钟，覆盖审计、四次收集、两次独立 reload 比较和结果保存；无训练、下载或 checkpoint 传输。启动成功后仅替换经身份核验的旧失败空闲监督器；旧 worker 进程组必须已退出。任何前提、finite、资源、reload 或验收失败即停，保留负结果，不自动开下一炉；完成或截止后使用原有带保护关机流程，不释放实例。

尚待实测：续接审计、B 两臂指标、A/B independent reload、最终比较与关机状态。原始训练源码身份为 `e11c7aaddc8c1eab84365d399ef8696cb5d0f83b`；评估源码以本文件所在提交及运行封存 hash 为准。
