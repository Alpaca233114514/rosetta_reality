# paired ABC：首步物理失败与命令分量定位

本轮在保存的 `bank-003b-scene-2445` 初态上，取得了 B/C 首步碰桌的双向干预证据：只把 `right_shoulder` 目标换回 teacher 值，B/C 的首步安全停止消失；只把这一维学生目标换入 teacher 命令，则 teacher 首步复现右夹爪两根手指碰桌。A 首步是两侧夹爪的反侧手指越限，多个臂目标参与，尚未找到单维修复。结论只针对该初态的一次 20 ms 控制间隔，不是多场景根因、策略修复、抓取成功或 M2 完成。

权威机器结果：`reports/training/paired-physical-command-localization-2026-10-04-002.json`，SHA-256 `de35b15fc58e41d5b0423b5ed8c905aae37e3dd5587f369a0d8afb1d222ce119`。001 版本保留。上轮全轨迹失败分型见 `paired-failure-localization-2026-10-03.md`。

## 证据与运行范围

用户要求继续排查并明确无需反复确认。本轮没有训练、下载、改权重、改安全阈值或电源操作。AutoDL 现有平台容器没有可见 GPU；所有新计算是 CPU 上的标准库检查或保存命令的物理回放，`nested_docker_used=false`。没有在 Windows 或 WSL 主机运行 ML/仿真。

使用已有 Python 3.12.3 环境，MuJoCo 3.8.1、gym-aloha 0.1.4、NumPy 2.2.6。系统 Python 没有 ML 包不表示此环境丢失。三臂 `pretrained_model` 是指向固定 step5000 checkpoint 的既有符号链接；普通文件枚举不能证明载荷缺失。本轮逐项读取并核验 manifest 内全部 39 个文件，三臂 SHA 全部一致，没有复制或加载策略权重。

新代码通过 `scripts/stage_autodl_from_wsl.sh --verified-source-delta` 登记到新的内容寻址 workspace；未覆盖旧 workspace。每个任务有独立 worker/guard，预算 180 秒、物理任务 RSS 2 GiB、证据 8 MiB；每个条件从同一积分状态独立恢复，最多走一个控制间隔，安全异常之后不接续第二步。

| 任务 | 实际完成 | 新策略 forward | 新控制步 | 接收器常规文件 |
| --- | --- | ---: | ---: | ---: |
| boundary-cpu-002 | 23 项源身份、13 项标准库合成测试 | 0 | 0 | 18 |
| saved-command-physics-001 | EGL 初始化失败，未进入回放 | 0 | 0 | 14 |
| saved-command-physics-002 | teacher/A/B/C 保存命令回放 | 0 | 4 | 20 |
| gripper-targets-001 | 4 原始控制 + 6 夹爪目标互逆 | 0 | 10 | 26 |
| arm-targets-001 | 4 原始控制 + 72 单关节目标互逆 | 0 | 76 | 92 |

共 90 个新物理控制步，0 策略 forward、0 optimizer update。170 个新收据文件按远端 manifest 逐项 SHA 校验，并用 PowerShell 第二算法复核精确文件集合与内部 manifest；这是同代理的算法核对，不称独立代理验证。所有已启动 worker/guard 均已退出；这些退出收据不证明平台停机或停止计费。

早期 boundary-cpu-001 在启动前被源身份校验拦住：远端旧 supervisor SHA 与登记的当前 SHA 不同。只在新 workspace 补齐登记版本，保留旧身份，未调整期望 SHA 来通过检查。物理回放 001 因无 GPU 的 EGL 初始化失败而停在 import；安装源码正式支持 `MUJOCO_GL=disable`，002 使用该选项且不渲染像素，没有安装替代后端。失败收据均保留。

## 物理回放先通过历史对照

`scripts/replay_saved_paired_commands.py` 读取保存的全 `mjSTATE_INTEGRATION`，包含 warmstart；验证积分 spec、compiled model SHA 和逻辑初态，调用当前安装任务的 `before_step`，经过原 10 个 MuJoCo 子步，再读取关节、接触和 reward。物理模型 SHA：`cea99c5e148cbfb5bf9c7139ecdf2a7ad92e872797808b3a6e61236eb6a56f78`。

teacher 首个纠偏命令来自源 trace 的 step178。其回放后的 293 项积分状态逐位复现历史值；A/B/C 的 14 维 `state_after` 和历史越限计数也全部逐位一致。后续两个分量实验再次复现四个控制的完整 qpos/qvel/ctrl/qacc/驱动力/约束力、关节明细、接触及 reward，三个阶段合计 12 个精确原始控制。

这里没有恢复完整 Python owners、回调状态或像素，也没有重新运行策略。数值对照支持解释这一控制间隔的物理结果，不能代替原生 CUDA observer parity、完整 observation 恢复认证或多步 rollout 等价证明。

## 三臂首步停止触发不同

| 保存命令 | 越限关节数 | 非预期接触 | 首步安全 |
| --- | ---: | --- | --- |
| teacher | 0 | 无 | 是 |
| A | 2 | 无 | 否 |
| B | 0 | 右夹爪两根手指与桌面 | 否 |
| C | 0 | 右夹爪两根手指与桌面 | 否 |

A 越限是 `vx300s_left/right_finger` 和 `vx300s_right/right_finger`。它们下限均为 -0.057，实测分别 -0.0570178207113、-0.0597151157288，超下限约 17.8 微米、2.715 毫米。原容差 1e-5 保留；未因左侧超限较小而放宽。

安装任务 `get_qpos` 只从每只夹爪的第一根手指计算 14 维逻辑开度，反侧手指不直接进入该状态。A 首步逻辑开度左 0.886970、右 0.730543，仍在 0–1 范围；这不能排除另一根手指越限。因此“命令未裁剪、逻辑开度正常”不能推出完整物理夹爪安全，也不能据此宣告安全检查误报。初态四根手指均在物理范围内。

## 只改夹爪目标不能解除首步停止

`scripts/replay_paired_gripper_targets.py` 只互换动作 index6/13，保持其余 12 个臂目标不变。

| 命令组合 | 越限数 | 非预期接触数 | 安全 |
| --- | ---: | ---: | --- |
| A 臂 + teacher 夹爪 | 1 | 0 | 否 |
| B 臂 + teacher 夹爪 | 0 | 2 | 否 |
| C 臂 + teacher 夹爪 | 0 | 2 | 否 |
| teacher 臂 + A 夹爪 | 0 | 0 | 是 |
| teacher 臂 + B 夹爪 | 0 | 0 | 是 |
| teacher 臂 + C 夹爪 | 0 | 0 | 是 |

A 的右侧反指仍越限约 1.846 毫米；B/C 的碰桌保持。学生夹爪目标并非在 teacher 臂目标下单独充分造成这些首步失败。改变夹爪目标会影响动力学，但不能将当前失败归结为一个通用的开度偏置，也不能只改 decoder 开度作为已证实修复。

## B/C 的 right_shoulder 具有双向首步因果证据

`scripts/replay_paired_arm_targets.py` 在四个精确原始控制之后，对 12 个臂目标分别进行单维 teacher→学生和学生→teacher 替换，共 72 个条件。每个条件其余 13 维保持原值，物理初态和控制间隔固定。

初始右肩 -0.20359555 rad。teacher 目标 -0.18546352；B/C 目标分别 -0.08987497/-0.08856781，目标位移约为 teacher 的 6.3 倍。

| 命令 | right_shoulder 目标 rad | 首步右肩实测 rad | 碰桌接触数 | 安全 |
| --- | ---: | ---: | ---: | --- |
| B 原命令 | -0.089875 | -0.069244 | 2 | 否 |
| B 只换该维为 teacher | -0.185464 | -0.199970 | 0 | 是 |
| teacher 只换该维为 B | -0.089875 | -0.099699 | 2 | 否 |
| C 原命令 | -0.088568 | -0.066567 | 2 | 否 |
| C 只换该维为 teacher | -0.185464 | -0.199970 | 0 | 是 |
| teacher 只换该维为 C | -0.088568 | -0.097977 | 2 | 否 |

在扫描的 12 个单维目标中，只有 index8 `right_shoulder` 的 teacher 替换使 B/C 首步完全安全；反向替换也准确产生原来的两根右手指碰桌。无需将相关性当因果，或将一维修复外推为整条任务修复：这是一处已由互逆干预定位的首步物理触发。

## A 存在多目标与双侧耦合，尚无单维修复

A 的全部 12 个单维 teacher 替换都没有让首步整体安全。换回 `left_wrist_rotate` 只消除左侧越限，右侧仍越限；换回 `right_waist` 只消除右侧，左侧仍越限。不能由这些部分结果推定两维合并一定通过，合并尚未测。

把 A 的单维目标放进 teacher 命令，`left_waist`、`left_forearm_roll`、`left_wrist_angle`、`left_wrist_rotate`、`right_waist`、`right_forearm_roll` 六项分别能够诱发一根手指越限，涉及不同手指。A 的右前臂目标相对初态跳变约 -2.680 rad，左前臂约 +1.873 rad，但跳变量大本身不证明单位或 ordering 错误。互逆结果支持多项臂目标可经动力学诱发夹爪越限，不能指定一个唯一坏维度。

## 下一层排查仍是实际推理边界

当前已区分“命令如何造成首步停止”和“模型为何生成该命令”。前者已有上述物理证据；后者没有新的策略 forward，尚不能在 observation/processor、噪声、离线学习、恢复分布或多步时序之间定唯一原因。teacher 目标使用特权几何信息，本轮替换不能成为在线控制补丁。

后续 CUDA pilot 入口为 `scripts/paired_boundary_pilot.py`，新计划 `configs/diagnostics/paired-boundary-pilot-20261004-002.json` 已绑定 13 项合成检查收据；仅证明采集器的标准库生命周期与拒绝路径，尚无真实 CUDA、完整像素恢复或 observer parity 通过证据。原封保留 A1/A2/A3、teacher、B、C 共最多 5 次 forward/6 个控制步、0 optimizer 的范围，先验 exact restore、A/A、observer parity，再比较真实 processor state/noise/raw/projected 输出。

重点转向 B/C `right_shoulder` 的实际输入到归一化、flow 输出和物理解码各段，及 A 多关节大目标与未观测反指的物理耦合。普通场景四条安全满时失败与 2447 接触保持失败仍未被本轮解释。正式 Gate 结果为 null，M2 未完成。

## 文件与复核

注册计划、源码 SHA、输入 SHA、资源预算和所有失败身份均保留于 `configs/diagnostics/` 与各接收器；公开报告只使用仓库相对路径。机器证据入口 `scripts/summarize_paired_physics_20261004.ps1` 重新检查 170 个收据文件、39 项 artifact 载荷核验记录、12 个精确控制及 72 条单维变更约束；`git diff --check` 通过。001 汇总与 002（增加总文件数）均保留。

新增诊断代码未改动历史训练/测量入口、Action Contract、归一化、loss、checkpoint 或安全限值。已有 staged/unstaged 工作保留；未 commit/push。
