# Q0 integration state 保存/恢复工程检查

状态：仅实现，未运行。入口 `scripts/check_q0_state_restore.py --output <new-receipt.json>`；必须在已登记 AutoDL 原生容器中，由独立 supervisor 设置至多 120 秒硬截止。入口自身只作 cooperative 时间检查，不能中断阻塞的 reset、render、native step 或 JSON 写入。只用已安装依赖，不下载模型/数据、不安装包。

固定 seed 10，一个环境一次 reset，两段各三步，最多六个 env.step。第一段每步使用当前 observation.robot_state 的标准 absolute joint position hold；第二段从保存状态重放完全相同三条 action。复用 `configs/sim/aloha_insertion_smolvla.yaml` 和原 adapter，越界 hold、clipping、done、关节限位或非预期碰撞均停止，不改变物理、奖励、bounds、阈值。零模型调用、零 optimizer、零 teacher、无 sealed seed 读取。

复用 `eval/reproducibility_capture.py:PhysicsReader` 的原生 model/data pointer、`mjSTATE_INTEGRATION` spec、完整 float64 state 和 model hash。恢复需要已安装 MuJoCo 的 `mj_setState`、`mj_forward`；先 setState、forward 重建派生量，再 setState 恢复 forward 可能改动的 warmstart/control，并验证 integration 完整精确回环。每一步比较完整 integration bytes、observation（含图像）、reward、done/info、contacts、原 adapter 的安全计数与 clip mask，不使用 qpos-only 冒充恢复。

PythonState 遍历 adapter/Gym wrapper/Gym-ALOHA/dm-control task 等可识别 Python owner，保存整个 __dict__、对象引用、数组、RNG、计数器和 Python/NumPy/Torch CPU 全局 RNG；恢复保留 owner 身份和跨 owner 的字段别名。实际遍历 owner 的已安装 Python 源文件 SHA 进入收据，完整 wrapper snapshot、RNG 数值、integration 数值及 bytes/hash 均保存。未知模块 owner、opaque state、stateful closure、无法检查的 native callback 或插件外部状态直接 unsupported。此支持边界刻意保守，远端安装的包装层若包含未支持类型可能零步即返回 unsupported，不能据此宣布恢复通过。

输出使用 exclusive create，已有文件不启动环境、不覆盖。先留 incomplete 收据；正常/失败结束在本次新文件写最终状态和完整 JSON SHA。数组保存 dtype/shape/完整 bytes 及 SHA，integration 和每步观测另存数值；120 秒硬杀时只保证最初 incomplete 收据，不保证中途六步全部持久化。异常只记录类型，不把任意带主机路径的异常消息写入日志。

源依赖：`src/rosetta_reality/eval/reproducibility_capture.py`、`src/rosetta_reality/sim/gym_aloha.py`、`src/rosetta_reality/sim/action_contract.py`、已安装 MuJoCo/NumPy/Torch/Gymnasium/Gym-ALOHA/dm-control。本地未检查远端实际安装源码，运行时 hash 和 fail-closed 检查才是环境证据；源码 hash 本身不证明恢复的数学完整性。`tests/test_q0_state_restore_stdlib.py` 只覆盖 create-only、unsupported、异常信息脱敏及 hash 灵敏性，本轮按要求未跑测试。

通过仅表明当前安装、seed 10 reset 附近三步 hold 的工程性精确重放；不能推广为 loaded-contact 全局确定性、teacher 正确、Gate 3/4、成功标签或 M2。尚无真实执行结果。

补充支持边界：源码已定义的 frozen slots `ActionContract`/`ActionDimension` 显式按 dataclass 全字段捕获、序列化与 hash，保留共享引用并在 restore 前核验未被变更；不以 unknown owner 忽略合同。Enum 保存类型、成员及值，函数/方法/类保存完整符号身份，dtype 保存精确描述。wrapper 收据的绝对路径字符串只保存 SHA256 与长度，内存恢复快照仍保存原值。每份输出固定 `provider_qualified=false`；`unsupported`/`mismatch` 为逻辑失败，`infrastructure_failure=false`，ImportError/Timeout/未预期异常为基础设施失败；原 adapter 安全停止用 `safety_stop=true` 和 `status=safety_stop` 要求停止全批次。新增基于实际 dataclass AST 的 stdlib 合同字段/共享引用测试，以及状态分类和路径脱敏测试，仍未执行。
