# Hestia：GPU 预检与两步更新观察器验收

状态：预登记。用户已开启带卡实例并明确要求利用 GPU；允许本分支提交、推送，结束后关机，不释放实例。本模块由独立 supervisor 守护，最多 1200 秒计算，结束或失败后最多 120 秒小证据回收窗口，然后执行已核验的平台关机入口。失败不自动重试。

## 问题与边界

上一轮 8→40 场景、固定 256 updates 的比较失败：开发视觉门槛两臂均 0/4；B 的训练关节拟合 4/4、夹爪仅 1/4，第一步物理误差回退。详见 `reports/training/m2-smolvla-native-visual-coverage40-result-2026-09-10.{md,json}`。M2 未完成。

新炉名称为 `m2-smolvla450m-visual-hestia-fit40-001`，保留历史 Athena 文档与失败记录。候选延续已经 CPU 核验的修订设计：40 个 frame-0 场景，fresh pinned base、batch 4、seed 20260809，1280 updates、1280 cosine decay、warmup 16；固定 native loss、VLM 冻结、expert-only、state projection、optimizer 与其余合同。这同时改变更新预算和学习率轨迹，不宣称纯重复次数的因果效应。

**本模块只执行真实预检和独立的两步 smoke，不执行候选 1280 步。** 新 B/C 评分器和观察器适配首先接受回归；完整 B/C 模型采集与候选训练完整性封存入口尚未完成，不以已有评分函数替代这项门禁。旧 A/B evaluator 的 256-step 身份断言保持不变。

## 固定身份与入口

- 源码来自当前获准推送的功能分支 commit，在新 clean detached workspace 执行；启动记录 commit、全部被跟踪 Python/YAML/JSON 文件及本计划 SHA256、实际 stage YAML SHA256。
- 基座 `c83c3163b8ca9b7e67c509fffd9121e66cb96205`；VLM `7b375e1b73b11138ff12fe22c8f2822d8fe03467`；数据 `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`。原生 LeRobot 源码与 sampler/scheduler hash 延续既有封存身份。
- 父配置、Action Contract、训练集 normalization、processor、split、模型冻结范围按 `configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml` 的不可变引用核验。实际训练 block 的预算/decay 改为 1280；smoke block 单独固定 2 updates，不续跑任何已有 optimizer。
- AutoDL 运行入口 `scripts/run_autodl.sh`，profile `configs/runtime/autodl_rtx4090.yaml`，`nested_docker_used=false`；离线，不下载模型/数据、不上传公开 Trackio。
- 每阶段名字 `m2-smolvla450m-visual-hestia-{preflight-b1,preflight-b4,smoke2}-001`；输出 `runs/hestia-gpu-preflight-001/`，checkpoint 使用注册 durable checkpoint root 下各自新名字。
- 实测 GPU 名称、UUID、软件版本、空闲空间由本次 registration/doctor 记录，不沿用历史关机或 GPU 身份。预检前保留预算仍要求至少 11,831,765,228 bytes；不删除旧证据。

## 执行顺序与验收

入口 `scripts/run_hestia_preflight.py`：`prepare` 仅核验/封存；`supervise` 创建外部期限及独立进程组，`worker` 依次执行：

1. Ruff；新 visual-fit/observed-launch 测试以及原 visual-coverage、observation、fixed-sample、tracking-composition、error-boundary 回归。pytest 不允许失败或跳过。
2. `check_env.py`、doctor、真实缓存 data tests、benchmark。doctor/benchmark 使用独立输出目录，避免复写旧证据。
3. 原 v2 launcher 的 batch 1、batch 4 no-optimizer forward；8 GiB allocated、10 GiB reserved、10 GiB host RSS 上限。
4. 原生固定 sampler 的独立次序检查；45 个非 hidden frame-0 的相机、非视觉条件、标准动作语义与 train-view 索引检查。hidden 仅使用已登记 ID 排除，不读取其样本。
5. fresh base 两步 smoke。`training/observed_launch.py` 在原 v2 launcher 外安装观察器；native trainer、feature stack、optimizer、scheduler 与 RNG 流程保留。观察实际交付 batch 和成功 optimizer updates，要求恰好 8 样本、2 成功更新；额外保留逐次 optimizer-step 前 finite gradient 检查。预取索引不冒充已完成更新。
6. 原 checkpoint 更新审计；两个独立进程对相同 45 场景、4 种固定噪声、50-step 全 chunk 生成 normalized/standard 输出，要求全部 dtype 与数组逐项精确相同，且进程 ID 不同。保存完整 checkpoint/processor/optimizer 状态与原始证据。

任何身份、回归、输入、finite、资源、时间、checkpoint 或 reload 失败均保留负结果并停止。失败后没有主训练自动回退。CPU 合成测试通过和 GPU smoke 通过均仅说明工程路径可用，不能称为视觉泛化修复或 M2 完成。

## 候选的后续验收（本模块不执行）

复用原 B，候选 C 使用新的 `frame0_all_nonself_mean_error_fit40_v1` 协议。固定噪声与全部非视觉输入，比较正确图与所有非自身错配图的平均误差，以及视觉无关均值下限；不能先平均预测。开发 5 episodes 已被多次使用，不称为独立测试；hidden 保持封存。

C 的训练拟合和开发评估都要求至少 3/4 **相同噪声条件**下 aggregate、关节、夹爪分别同时满足正确图更好且低于均值下限。B 保留原 aggregate 训练拟合门槛，其已知夹爪负结果不作为阻断 C 的新前提。C 相对 B 的开发改善及 standard 全 chunk/first-action 关节/夹爪非回退门槛沿用原公式；独立 reload 要求七个完整数组完全一致。固定末步 1280，不依据开发结果挑选 checkpoint。

候选还必须具有实际 saved config/processor inventory、冻结与更新 tensor 审计、5120 个成功消费样本及全部 LR、checkpoint 完整性、独立模型采集封存证据。未完成之前不执行候选；失败后保留结果，不自动换损失、dropout、unfreeze 或开下一炉。
