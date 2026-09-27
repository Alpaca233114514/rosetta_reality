# 根因诊断设施后续修复（2026-09-22）

本轮修复评分、草稿预检和资源执行边界。没有新增真实模型、CUDA、训练或 rollout
结果；唯一学习侧根因仍不可识别，canonical Gate 4 保持 0/5，M2 未完成。

## 修复与验证

- 保存预测的评分器拒绝四噪声下不一致的 target、mask、horizon 和动作维度；
  partial cohort 必须显式非空。包括 padding 在内的所有预测/监督值均须 finite。
- 开启与关闭事件分别匹配；完整窗口未发生与尾部截断分别记录，截断窗口不宣称
  持续保持。评分只证明数字集合一致，明确 `identity_verified=false`。
- smoke/A/B 草稿在保存前验证结构及已声明的 v2/VLA 源码 SHA。smoke 的实际
  2 次更新与 500 步 scheduler 参考配置分开；A/B 预算仍为 500/2000，独立原始
  base、train40 normalization、四轮覆盖，训练授权均为 false。
- runtime observer 使用真实原生 optimizer factory/update、Accelerate、FeatureStack
  和输入/更新观察器进行 CPU 合成集成验证。错误 loss 分母在更新前拒绝；
  4 次样本曝光对应 2 个唯一帧。测试使用单参数合成 policy，不是 SmolVLA 验收。
- runtime hook 被其他代码改动时，仍恢复自己拥有的另一个 hook，不覆盖外来修改。
- 离线分析读取数组前核对 cgroup 资源。plan 003 实测 1536 MiB、2 CPU，使用
  180 秒命令超时，TERM 后 5 秒 KILL。plan 002 的旧 3 GiB 运行记录完整保留。

相关回归测试 **96 项通过**，Ruff 通过。测试含既有短暂成功、trace 状态不一致、
A/A 分叉拒绝归因，以及新增评分/资源/实际配置反例。CPU XPU 数量警告和既有
DataLoader fork 警告均记录在日志中；无测试失败。首次新评分测试的异常消息匹配
失败保存在 `test-001.log`，修复后测试通过，没有覆盖失败证据。

## 数值结果与解释

`root-neighborhoods-20260922-003` 的 metrics.csv、events.jsonl、neighbors.jsonl
与 002 逐字节 SHA 一致。独立复算通过 138510 项数值/邻居检查和 17280 项事件检查，
没有调用生产指标 helper。本次资源修复没有改变科学结论。

frame 125 模型仍优于两个简单邻域读出，局部后期退化不能直接归因于标签错误；
frame 250 保持幅值仍未击败训练常数基线。下一项能区分解释的真实实验仍是完整轨迹
可学性探针及 seed 1002 新三臂复现；本轮没有执行它们，也无法补回旧 004 缺失轨迹。

## 交付与后续边界

- 分析：`runs/root-neighborhoods-20260922-003/`。
- 独立复算：`runs/root-neighborhoods-independent-20260922-003/`。
- 新草稿：`runs/trajectory-probe-prepared-20260922-003/`。
- 原三臂草稿：`runs/root-evidence-20260922-002/reproducibility/`。
- 测试、SHA 比对和交付清单：`runs/root-diagnosis-followup-20260922-001/`。

历史 001/002 包及原报告保留。新草稿不能直接启动；未来还需真实模型/processor
集成、运行环境与授权窗口、watchdog 和完整依赖身份的独立登记。
当前本地修复不需要 SSH，真实远端检查开始前再请求连接信息。

用户说明 Basin MCP 可供查训练历史，但本会话的可调用工具、MCP 资源和模板中
尚未发现 Basin，因此没有声称已查询或导入。后续联合开发应按 run ID 关联
代码/config/模型/processor/数据/产物 SHA，同时保留记录来源和验证等级。
没有修改 Basin、提交、推送或启动外部计算。
