# Q0 candidate 输入拒绝边界工程检查预登记

状态：实现待执行；SSH 关闭期间不启动远端任务。候选
`mink-coupled-translation-teacher-20260927-001` 已知 G1 失败，仍未取得
provider qualification。本检查是独立 synthetic 工程 ledger，不是正式 G4
pass，不改变原 Gate 协议或合格标准。

入口：`scripts/check_q0_refusal_boundaries.py --output <new-ledger.json>
--max-seconds 120`；测试入口：`pytest -q tests/test_q0_refusal_boundaries.py`。
均由上层一次上机编排在登记 Linux 容器内启动；外层应同时设置 120 秒
进程硬超时，超时保留已有日志，不重用旧输出身份。脚本内部使用 monotonic
预算检查，不能替代外层硬超时。

固定 inventory 在脚本 `BOUNDARIES` 中：16 个正负同构对、32 cases，每个
case 独立重建两次，总计最多 64 次 observation/decision attempts。构造失败
不调用 decision：robot state NaN/shape、reward NaN、pose position NaN/shape、
quaternion shape/zero、缺 robot_state、negative collision count；其判据直接
来自 `InsertionGeometry` / `GeometryPose` 构造合同。decision 负例是已报告
异常碰撞及现有 socket/peg xyz 工作域六个边界，预期 reason 分别为
`unsafe_state` / `workspace_violation`。既有验证把 float32 pose 与 Python
float 阈值比较，0.55 的 float32 舍入可越过上界；因此工作域正例位于已有
边界内 1e-6 m，负例越界 0.01 m，真实阈值不变；lower 检查 socket，upper 检查 peg。

有效控制从初始关闭夹爪状态进入原有 HOLD 分支；使用 `NoSolveStub`，调用
solver 必须成为失败。这保留候选真实 `decide`、父类输入验证及 HOLD 逻辑，
没有 simulator step、model/data load 或 real IK，也不覆盖 INSERT lattice
求解、其他 stage/history 的边界或完整动作物理语义。不得由此推断 real IK、
闭环、安全任务成功或候选可担任 provider。

ledger 区分 action、explicit refusal、预期构造层清晰异常与意外异常。
仅确切 layer、预登记 built-in exception type 与既有合同错误文字签名匹配算清晰异常；所有其他
异常均失败，negative silent action 单独标记。异常仅保存 type，不保存
message/traceback 或主机路径。记录 inventory/source SHA256、预算与两次
重建输出；输出 create-only。构造层拒绝不能冒充候选 explicit refusal。

批处理字段 `status` 为 `passed` 或 `check_failed`，只对应工程检查结果，
`infrastructure_failure=false`。Timeout/启动未预期异常不写成功 ledger，
缺输出由上层停止批处理；逻辑 negative 留下 ledger，允许后续独立检查。

来源：`src/rosetta_reality/sim/teacher_gate.py`、`geometry_teacher.py`、
`scripted_teacher.py`、`geometric_teacher.py`、`coupled_translation_teacher.py`。
既有源码与历史证据不变；本预登记没有新增未定义的 observation 判据。
