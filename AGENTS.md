# Rosetta Reality — Agent 约定

## 修改边界

- 保留用户已有修改、实验数据和失败证据；不擅自删除或不可恢复覆盖。未经明确请求不提交或推送。
- 开发在功能分支进行；保护 `main` / `master`，不绕过 Codex Auto Review、PR 审核或必需检查。
- 仓库文档、日志和发布内容使用仓库相对路径，不写入主机绝对路径、用户名或凭证。

## 环境与授权

- PowerShell 可用于编辑、Git 和不加载模型/数据的静态检查；其他命令行使用 Bash，不使用 cmd.exe 或 Windows Python。
- 本地 ML 依赖、数据准备、模型、ML 测试、训练、评估和仿真必须从 `wsl.exe bash` 启动并在 Linux Docker 容器中执行。WSL 主机只作只读检查、源码检出、容器编排及受控 Hub 控制面操作。
- AutoDL 平台容器是唯一已登记远端例外，不嵌套 Docker；使用 `configs/runtime/autodl_rtx4090.yaml` 与 `scripts/run_autodl.sh` 记录环境身份。
- 本地按 16 GB 系统内存预算设计。不得擅自安装/升级系统 PyTorch、CUDA、驱动或仿真器，不靠扩大 swap、内存、反复 OOM 或无控 offload 强行运行。
- 模型/数据下载、外部 GPU 计算、正式训练、真机操作和对外发布须有明确授权。已登录 Hub CLI 不代表获得下载或发布授权，不读取或输出 token。

## 架构与证据入口

涉及 SmolVLA 数据、`configs/vla/`、`src/rosetta_reality/vla/`、trainer/loss/optimizer/scheduler、checkpoint/resume、export、validation 或 Gate 3/4 时，先完整阅读：

1. `docs/m2-smolvla-architecture.md`：组件与控制流的稳定入口。
2. `reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md` 及同名 JSON：审计框架与修复顺序。
3. `reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md` 及同名 JSON，以及架构文档指向的对应实验最新完成报告。

文档与 hash-bound config、Action Contract、可执行断言或不可变证据冲突时先核对，不修改历史证据来消除矛盾。组件入口、合同、控制流或 Gate 状态改变时，同步更新架构文档；稳定路径改变时同时更新本文件、`README.md` 和 `docs/architecture.md`。

截至 2026-09-11，Faust、Aster、Way、Zen-uniform (411)、Zen-firstaction (422)、vcdropout (433)、vfunfreeze (444) 的 Gate 4 均为 0/5，M2 未完成。最新学习侧闭环结论见 `reports/training/m2-smolvla-vfunfreeze-gate34-closure-2026-09-06.md`；不得以离线 MAE、梯度视觉敏感性或 frame-0 对齐通过推定闭环成功。新一炉须重新登记单变量计划并获授权。AutoDL 关机记录不是当前状态证明；实例不得擅自释放。

Hestia 已完成 1280 步，训练各组 4/4、开发各组 0/4（四个固定推理噪声条件），没有新 Gate 3/4。主实验权威见 `reports/training/m2-smolvla-hestia-main-recovery-result-2026-09-11.{md,json}`；四 checkpoint 与 CUDA 曲线均已取回。原生前缀采集已完成，180 次输出精确复现 C、参数未变、结果已回传，见同目录 `m2-smolvla-hestia-prefix-path-closure-2026-09-12.{md,json}`。最新本地八臂读出见 `m2-smolvla-hestia-prefix-readout-result-2026-09-12.{md,json}`：投影后物体位置仍可读，但开合时机和完整动作仍失败，不能凭探针定位唯一根因。互逆 K/V 参数检验 002 已完成 720 次 CUDA 推理，两个原生端点精确复现；22 文件已校验回传，见同日期 parameter-crossover-result 报告。16 个 K/V 更新对后期退化有双向因果贡献（左关节正向恢复 74.7%、互逆损伤 89.7%），但 hybrid 仍未击败相关常数基线，唯一根因未确定。旧入口失败和空接收器保留；参数联合检验 002 的关机请求已恢复，见同日期 parameter-crossover-shutdown-recovery.json，不以请求记录证明当前计费状态。K-only/V-only 六组检验也已完成 1080 次推理、30 文件回传校验和 69,440 项独立数值核验，见同日期 kv-split-result 报告；V-only 对左关节后期退化的恢复/互逆损伤为 59.1%/69.7%，K-only 为 19.5%/13.8%，但仍未修复泛化。本轮关机请求及匹配收据已取回。逐层 V 互逆已完成 18 条件、3240 forwards、78 文件回传与 193,664 个独立数值核验，见同日期 v-layer-result 报告。退化分布于多个 V 层，第 7 层左关节恢复 17.52%，并非唯一坏层；场景协方差与整体偏移均参与，夹爪时序仍失败。逐层实验关机请求及匹配收据已补回，见同日期 v-layer-shutdown-recovery.json，平台也实测关机；后续已授权开机运行新诊断。用户已授权根据结果连续排查，无需再询问 SSH 窗口；分量诊断首个 forward 因把 64 个相机 token 切片误作完整前缀而阻断，8 文件已回传；见同日期 value-component-failure.json。源码审查确认完整前缀为 3×64 图像 +48 语言 +1 状态 token，state_proj 两参数变化，不能将冻结 VLM 误作整段前缀不变。value-route 已完成十条件/1800 forwards/0 optimizer、50 文件回传和 95,744 项独立重算，见同日期 value-route-result；图像路由左关节恢复/损伤为 47.7%/50.8%，语言 13.1%/9.4%，状态 1.4%/3.4%，图像路由是主要贡献位置，唯一根因仍未定。image-value-component 已在 20260912T151222Z-bd007f5402bf-46803787f6a4 完成十条件/1800 forwards/0 optimizer，耗时1195.953秒；52文件回传校验、95,744项独立复算通过。只在真实图像64-token拆分共同均值、token模式及场景残差，完整241-token GEMM和其他输出保留；四个对照共720次输出精确复现。共同均值解释左关节640→1280退化的35.9%恢复/33.8%互逆损伤，部分场景残差更新对右关节和夹爪有益；640时已存在的泛化差距仍未解释。见同日期 image-value-component-result。平台已再次实测关机，本轮shutdown-request待下次窗口只读恢复；value-route请求已补回。用户最新要求一小时内收尾、明天接续新对话；今晚停止计算，接续步骤见 m2-smolvla-hestia-next-session-handoff-2026-09-12.md。使用各自新身份，切勿重跑旧任务。Oracle 使用答案，FK 是命令位姿而非实际到达位姿；未修复泛化或选新 checkpoint，不得将旧 pending 当作重训依据。

## 模型与数据合同

- ER / System-2 使用 Qwen3.5；VLA / System-1 使用 revision-pinned `lerobot/smolvla_base` 450M。两者的训练、checkpoint、optimizer、特征和 artifact 分开管理。
- 旧 Qwen frozen-feature + action-head VLA 实验只作历史负结果；可复用经核验的数据身份、split、Action Contract 和仿真协议，不可复用权重/特征初始化 SmolVLA，或当作 ER 能力证据。
- ER/VLA 通过版本化 structured `ActionPlan` 连接，至少含 `subtask`、`object`、`target`、`motion_hint`、`constraints`、`success_condition`、`replan_condition`。M3 须先分别通过 M2 闭环与 ER 独立验收，再验证 integration。
- 数据层保持 embodiment-agnostic，policy 通过 simulation adapter 控制 MuJoCo；不把模型或仿真专用逻辑泄漏到通用接口。import 不加载模型、访问网络或修改环境。
- Action Contract 明确 shape/dtype/range、每维语义与 ordering、单位/坐标系、absolute/delta、控制模式、频率/时间对齐、chunk 执行策略、gripper 编码和限幅。不得仅凭 tensor shape 判兼容，或硬编码动作维度、设备和 dtype。
- M2 优先复用固定 revision 的 ALOHA insertion 50 episodes 及既定 split；当前容器须核对 manifest/checksum、camera/state/action/task、fps 和 chunk 语义。不得仅凭 repo_id 复用；按 episode/trajectory/scene 防止数据泄漏。
- Feature Cache 身份包含模型/数据/processor revision、sample/camera、预处理/prompt、提取层/pooling、dtype/量化与 schema。身份变化须新建 cache，不静默复用或覆盖。LoRA/full fine-tuning 不复用冻结特征；SmolVLA 使用原生 online processor。
- Qwen ER 0.8B → 9B 复用流程和合同，不跨 backbone 复用特征或默认迁移 projector；warm start 须受控比较。9B 训练只在获批且实测资源通过的 GPU 环境进行。

## 模型工作门禁

先冻结假设、单一实验轴、代码/模型/数据/processor 身份、Action Contract、配置、资源预算、验收和停止条件。仅运行当前任务授权的阶段，不因后续流程列在此处就自动启动。

1. 不加载权重和数据的静态/schema/contract 检查。
2. 在 revision/digest 固定的容器内运行 `python scripts/check_env.py`、相关单测、dummy forward、CPU/小模型 smoke；真实缓存还需只读 inspect 和 `pytest -m data`。
3. Gate 1：确定性小幅 scripted action 验证 joint/actuator 顺序、方向、单位、gripper 和限幅。Gate 2：expert dataset replay 验证完整物理语义。失败不得训练或 policy rollout。
4. 新训练能力从 batch 1、极少步、固定小样本开始，核验 processor、normalization、forward/backward、梯度、资源、checkpoint/resume 和 Trackio；通过小数据 overfit 与短 GPU smoke 后才能扩大。非 finite、OOM 或 accelerator 不受支持时停止。
5. 正式 training/validation、validation-only selection、export 和 independent reload；保留完整 provenance 与失败结果。
6. Gate 3：短闭环检查 finite、合法 action、state 更新、完整 adapter、稳定性和 reload 一致性。Gate 4：固定协议评估 task success、rollout length、invalid action、joint-limit、collision、smoothness 和推理/仿真延迟。

M2 必须有可追溯、可导出、独立 reload 的 checkpoint，且通过注册闭环任务验收；能够生成指标或动作张量不足以完成 M2。ER 独立报告 plan quality，VLA 报 execution quality，integration 报 end-to-end success。

## AutoDL 运行合同

- 本地负责编辑、分析和决策，AutoDL 仅作获批 CUDA worker。开机前固定唯一 run name、代码/config/plan checksum、缓存身份、optimizer/scheduler、batch/steps、benchmark/smoke 命令、输出、验收/停止条件和预计时长。
- 传输使用已授权推送的功能分支 commit；未获 commit/push 授权时使用 `scripts/stage_autodl_from_wsl.sh` 创建 versioned content-addressed workspace。不手工覆盖远端代码，不用 `--delete`，同一 run 不跨 workspace identity。
- durable cache 保留 revision 与 manifest，不因缺缓存擅自下载。先 doctor 核验身份，再 benchmark、no-optimizer CUDA forward、两步 optimizer smoke，最后按单独登记计划正式训练；记录 `nested_docker_used=false`。
- 长进程必须 tmux 或等价守护，日志落 durable storage，SSH 断开不影响生命周期。启动检查进程/GPU、首批 step、finite loss/gradient、显存、Trackio 和 checkpoint。
- 稳定训练后控制 shell 使用 Bash `sleep 300`，每五分钟只采样一次最小状态，不高频轮询训练或连续 tail。等待可异步让出交互；不得把 sleep 注入训练进程。完整审计仅在健康检查、失败、注册 quarter/checkpoint 或完成边界进行；确认退出后不再启动 sleep。
- 完成顺序：训练退出并保存 → validation-only selection → export → 远端 independent reload → 回传 selected artifact/manifest/metrics → 本地 checksum/reload → 在授权范围内关闭计费 GPU → 本地 Gate 3/4 → 分析。失败保留远端状态，不直接开下一炉。
- 黑匣子保留 resolved config/plan、代码/cache/环境/GPU 身份、Action Contract、optimizer/scheduler、log/metrics/Trackio、checkpoint/selection/export manifest、人工介入和退出状态。完整恢复 checkpoint 默认保留；释放或迁移前须备份到用户批准的可靠位置，远端本地盘不能是唯一副本。

## 记录与发布

- 正式 run 预先固定 Trackio project/Space/run name/config identity/durable store。Space 不可用时保留本地指标，可按授权在 checkpoint 边界脱敏同步，不冒充实时。
- 公开 Trackio 仅同步经审查的指标、非敏感超参数、不可变 revision、资源与 run 状态；不上传凭证、主机路径、原始样本、对话、完整日志、checkpoint 或未审查媒体。
- 权重、数据、Feature Cache 和大型生成物不进 Git。保留配置、版本、seed/split、协议、环境、人工介入及负结果；不以 proxy metric 代替真实任务验收。
- 对外发布须明确目标仓库与版本授权、许可证检查、Model Card/输入输出合同与独立公开 reload 验证。ER/VLA artifact 不混用；优先发布实际修改的 policy/head/adapter，完整权重须另获授权。未验证真机时标注 experimental / research only。
