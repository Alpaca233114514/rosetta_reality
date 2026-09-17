# Luna 后训练改动审查

结论：当前版本不具备 CUDA/Gate 启动条件。Luna 已暂停修改和远端执行。
审查对象是新增 endpoint、offline、sim_gate、posttrain validator、shell runner、
Gate YAML、后训练 JSON/MD 和对应测试；不是将仓库全部未提交修改归给 Luna。
训练及回收结果不因这些尚未执行的新代码缺陷而被改判失败。

## 发现

1. **P1 — 导出配置缺少必需适配字段。**
   `scripts/canonical_fullframes_endpoint.py:149` 直接从原 YAML 的
   `model.action_space` 构造 artifact，但 `adapt_to_pi_aloha` 在
   `model.policy` 下，没有并入。offline 的 `SmolVLAActionSpace(**config["action_space"])`
   和选用的 Way Gate loader 都会抛 TypeError。应使用现有显式 Action Space
   resolver 序列化完整合同，同时解析继承配置，不另造不完整字典。
   纯标准库提取真实 dataclass 和父 YAML 字段的反例已复现该 TypeError。

2. **P1 — baseline 先读取隐藏行再筛选。**
   `scripts/canonical_fullframes_offline.py:157` 对整个 parquet 做 read_table，
   并先将 action 列转为 Python 列表，再判断 episode 是否属于训练集。
   `prepare_smolvla_train_stats.py:109` 的 view 复制原始数据，只替换统计量，
   元数据仍为 50 episodes / 25000 frames；它不是训练行专用的数据切片。
   因此 hidden=false 的报告声明与执行不符。应在 Arrow 查询/扫描边界以
   train episode predicate 筛选，再物化 action，并添加隐藏行哨兵反例。

3. **P1 — 验证的协议与实际执行的 Gate YAML 脱节。**
   `scripts/canonical_fullframes_sim_gate.py:58` 验证的是后训练 JSON；
   `_load_artifact` 没有将实际 YAML 的 gate3/gate4 阈值、seed、步数与它比较。
   实际引擎 `smolvla_sim_gate.py:616`、`:719` 使用该 YAML 执行和判断。
   修改 YAML 的 minimum_task_success_rate 或 seeds 可绕开 JSON 的固定值检查。
   当前 implementation 清单也不包含该 YAML。应单独封存实际渲染后的计划，
   并逐项验证其协议，添加真实 Gate-loader 的篡改反例。

4. **P1 — 新执行链未经重载一致性验证，却沿用旧通过标记。**
   `scripts/canonical_fullframes_endpoint.py:295` 在仅做文件准备后直接填入
   reload verified/exact=true。既有证明属于 `iris_runtime.load_native` 的
   保存 processor 加载路径，仅覆盖当时的八帧。新 offline 在 `:97` 明确
   覆盖 normalizer/unnormalizer 的 stats，新 Gate 则改用旧 Way loader，
   两条都不是旧证明实际跑过的路径。endpoint 只核验模型权重 SHA、检查
   processor 文件存在，再给当前文件重新生成 manifest，没有逐文件与已备份
   checkpoint seal 绑定；Gate 也没有核对 manifest 中模型条目的 SHA 等于
   注册终点，只检查独立的声明字段。因此漂移的 processor/清单可被重新盖章。
   应核验完整已恢复清单，保留保存的 processor 状态，在新 offline/Gate
   adapter 上实际比较完整输入/输出后生成新证明，完成前保持未验收。

5. **P1 — 声明的运行时限和资源保护没有执行器。**
   `scripts/run_canonical_fullframes_posttrain.sh:98` 等入口直接启动前台 Python；
   没有实现计划所述 3600 秒工作 deadline、4200 秒保护关机、进程树/RSS/CUDA
   守护，也没有失败 finally 收尾。磁盘/output cap 仅在阶段前后检查。
   这不是已有训练 supervisor 的子阶段。必须提供并验证独立生命周期执行器，
   不能把 JSON 中的预算常量当作已经执行的限制。

6. **P2 — internal_gripper_support 实际混入全部关节。**
   `scripts/canonical_fullframes_offline.py:253` 对 last_model_action 的全部
   50×14 元素直接 flatten；processor.py:319 证明该属性是完整动作，不是
   夹爪切片。因此 reported min/max/sample count 不代表夹爪 latent support。
   应按 Action Contract 识别左右夹爪，分别计数并比较已登记的内部支持范围。

7. **P2 — inference latency 混入额外固定流损失 forward。**
   `scripts/canonical_fullframes_offline.py:233` 的同一计时窗口同时包含
   predict_action_chunk 和 policy(batch, noise, time)；后者是离线诊断额外开销。
   当前 inference_latency 字段不能与 Gate 的推理耗时直接比较。
   应分别计时预测、诊断 loss 和后处理，并明确边界。

## 其他启动前限制

- runner 和 Gate YAML 使用共享 experiment 目录及 suffix 001，而非本炉独立
  Gate 结果目录；存在历史输出碰撞风险，cap 也会计入历史文件。应新建本炉
  命名空间，不能清理历史文件来让脚本通过。
- 新测试文件仅覆盖 plan/schedule/threshold 等静态断言，没有真实 endpoint
  构造、processor reload、模型 Gate bridge 或上述字段错位的回归。
- 代码中的 CPU 图像除法当前可与 canonical 定义一致，不能仅凭它没有调用
  helper 就断言数值错误；但仍需显式 float32、byte recipe 和新入口等价证明。

## 验证范围

本次进行源码/配置交叉审查，读取真实已恢复证据与相关历史运行入口；
执行 `runs/canonical-furnace-preparation-20260914-001/review-luna-static.py`，
仅使用 WSL Python 标准库，未加载 torch、模型或真实样本。反例输出：
`SmolVLAActionSpace.__init__() missing 1 required positional argument: 'adapt_to_pi_aloha'`。
AST 检查同时确认新 Gate loader 未读取实际 plan 的 gate3/gate4 字段。
未启动新的 CUDA、离线评估或 Gate，没有修改被审查的实现和历史证据。
