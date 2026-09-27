# 双时间轴诊断 API

`rosetta.dual_axis.v1` 将训练成功更新、尝试更新、探针样本/噪声、rollout 控制步和
模型调用坐标分开。Basin 只读取采集证据，不执行模型、训练或仿真。

新增三个默认只读工具：

- `basin_timeline(run_id, axis?, start?, end?, offset?, limit?, field?)`：语义坐标过滤、
  分页事件与已声明/观测/缺失测点。start/end 为训练 update 或执行 rollout_step，
  field 为 values 下 JSON Pointer；完整样本覆盖不能只由测点存在推定。
- `basin_locate_deviation(run_id, axis?, metric?, section?, offset?, limit?)`：区分相对
  退化、绝对失败、缺失和累计趋势。section 为 deviations/absolute_failures/gaps/trends。
  20% 相对增加与 5 倍 A/A 底噪取较大者；锁定参照，二次超限确认、恢复单列。
- `basin_branch_compare(left, right, offset?, limit?)`：按显式坐标比较摘要与数据身份，
  给出不兼容项和首个证据差异，不将研究继承视为权重继承或证明根因。

配置 `--source alias=/directory` 才暴露 `basin_import_dual_axis(source,path,run_id)`。
导入流式核对所有上游文件；仅复制 metadata 和事件 shards，权重/大张量外置。
不反序列化 pickle。一个 shard 最多 32 MiB，单个外部 artifact 最多 4 GiB，
旧安全根和 create-only 约束保持不变。中断导入缺少 manifest，不能被当作完整运行。

外部数据按需使用既有 `basin_read_artifact` 的 source/byte_offset/byte_limit 参数：
先核对记录内的 artifact 声明和受控来源，再重新流式校验完整 SHA，返回最多 4096
字节的 base64 切片。history 的 `verify` 不等价于外部文件当前可用性证明。

新记录存 `event_shards` 和 `event_count`，语义事件用新接口查询。旧 adapter 的
events/analyze/compare 行为保持兼容。API 回包仍受 128 KiB 限制，超限需缩小 field
或分页，绝不静默截断成“完整”证据。默认目录现在提供 12 个只读工具；登记来源后
共 16 个工具。CLI `call`、Python `BasinAPI.call` 与 MCP 使用同一个分发实现。

缺失 A/A 底噪或输入/noise/processor/Action Contract/runtime 身份不匹配时拒绝归因。
未采集值不是零；未越过告警阈值不是模型正确或 Gate 通过。累计缓慢退化作为描述性
趋势报告，不冒充相邻测点告警。所有因果结论默认 null。
