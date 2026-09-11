# 全局预测时移诊断

登记 `chunk-time-shift-20260911-001`。独立检验共同误差是否主要是整个预测
轨迹的统一时间错位。复用原零噪声native/projected-target数组，不读新数据或模型。

固定单个整数lag∈[-10,10]（每格20ms，最多±200ms），对所有场景、动作组和
槽位相同。处理 `P_shift[t]=P[clip(t+lag,0,49)]`，边界保持端点，不丢弃任一
目标槽位，不平移真实标签。负lag会保持首动作不变，不能由后续chunk改善推定
当前first-action执行策略改变。

仅40train按完整chunk混合单位MAE选择lag；同分优先绝对值小，再优先负值。
dev5不选lag；校正LOO逐行只用其余39行选择。关节和夹爪各自的train最优lag
仅作冲突诊断，不作用于dev候选。若选到网格边界，保留结果不扩展。

主判据与共同偏差诊断对应：两组full dev MAE均比控制降低至少20%、低于
两个train常数、正确图收益为正；LOO低于对应两个常数；首动作MAE不退化。
均使用原1e-8容差和相同full/first/early/middle/late/last窗口。
整体未满足就不把全局时移认作充分解释。通过也不证明数据错标或允许部署。

CPU已有Linux Docker，2核、内存/memory+swap各2GiB、离线、180秒上限。
在XPU噪声运行退出后执行，避免叠加资源。先check_env、Ruff和合成隔离检查，
输入/code SHA通过后一次执行。create-only保留数组、LOO、grid分数和完整指标，
exact array reload。无权重修改、optimizer、SSH或Gate；M2未完成。
