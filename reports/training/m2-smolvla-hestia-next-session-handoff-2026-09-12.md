# Hestia 视觉泛化排查：今晚收尾与次日接续

用户最新安排为一小时内收尾、明天继续，并要求完成后新开对话接续。今晚计算已结束；目标实例在本轮完成后的平台界面显示已关机。唯一根因尚未确定，目标继续保留。没有新训练、checkpoint 选择或 Gate 3/4；M2 未完成。

已提交创建接续任务“Hestia 视觉泛化根因续查”，初始消息要求只接收交接、今晚不计算。已创建单次次日上午 09:00 接续自动化 `hestia`，在来源任务醒来后找到新任务并发送一次继续提示；创建时新任务仍处于异步准备状态，不把排队状态当作已开始执行。

## 已完成结果

阅读顺序：当前 `AGENTS.md`、`docs/m2-smolvla-architecture.md` 及其必读审计，再读以下同目录结果。旧失败与 pending 仅作历史，不能当作当前待执行队列。

| 实验 | 已完成推理 | 回传文件 | 核心结论 |
|---|---:|---:|---|
| parameter-crossover 002 | 720 | 22 | 16 个 K/V 更新对后期退化有双向因果贡献 |
| kv-split | 1080 | 30 | 左关节 V-only 恢复/损伤 59.1%/69.7%，高于 K-only |
| v-layer | 3240 | 78 | 多层分布，第 7 层贡献最大但不是唯一坏层 |
| value-route | 1800 | 50 | 真实图像位置贡献最大；语言和状态位置也有视觉上下文 |
| image-value-component | 1800 | 52 | 共同图像 V 偏移是已测后期退化的主要有害分量；部分场景残差更新有益 |

上述实验均 0 optimizer。对应权威报告为 `m2-smolvla-hestia-{parameter-crossover,kv-split,v-layer,value-route,image-value-component}-result-2026-09-12.{md,json}`。最后两轮分别通过 95,744 项独立重算，逐层轮通过 193,664 项。不要把它们的分母、阳性对照或文件清单混用。

最新共同偏移在左关节上的恢复/互逆损伤为原 640→1280 退化的 35.9%/33.8%；这不等于解释了全部泛化差距。左关节原生 640 MAE 为 0.058508，原生 1280 为 0.061673，共同均值回滚后为 0.060538，train mean 常数为 0.047658 rad。唯一根因仍然开放。

完整前缀为 241 位置：真实图像 64、空相机 128、语言 48、状态 1；有效位置共 73。真实图像输入在两 checkpoint 的八个奇数层均 45/45 行精确相同，但训练过的 state_proj 会改变状态前缀。整段语言哈希含 padding，不能据此推定有效语言改变。不要重复首轮将 64 token 切片误作完整前缀的错误；该失败的 8 个文件和所有旧接收器均保留。

## 次日第一个排查轴

先解释 320/640 时已经存在的泛化差距。停止继续把 640→1280 的局部因果解释当作整个问题的根因。优先只读分析已核验数组，不开 GPU：

1. 从四 checkpoint 的原生 CUDA 数组核对 checkpoint、target、split、四噪声及 normalization 身份，复用完成曲线闭合报告。
2. 登记一个新的、未使用的分析身份，区分场景平均动作偏差、预测场景方差与预测/目标场景协方差。相对训练标签常数的 MSE 差应由这三项精确闭合；按 320/640、关节/夹爪、完整/首动作及四噪声展示，不挑开发集最优超参数。
3. 可预先固定一个仅用 train40 模型输出均值的场景无关模板作为输出层诊断。它不使用开发集答案，但也不是重新执行的策略，不能宣称修复或闭环成功。该分析尚未创建、尚未执行。
4. 根据实际结果再冻结下一轮单轴互逆检验。若需要比较更早 checkpoint 的 state_proj、其余 action-path 或 V 路径，先做本地参数身份审查。不要默认沿用640/1280 donor 缓存，不新增训练，不枚举未注册大网格。

早期曲线权威是 `m2-smolvla-hestia-cuda-checkpoint-curve-closure-2026-09-12.{md,json}`；旧 `cuda-checkpoint-curve-result-2026-09-11.md` 是部署失败记录，不是当前已完成曲线。四步 320/640/960/1280 的完整开发关节与夹爪都没有同时击败两个常数基线。640 是已测最低开发均值，不是通过验证的选择。

## 证据入口与续跑边界

- 四步原生数组：`runs/hestia-cuda-curve-recovered-20260912-001/{000320,000640,000960,001280}/arrays.npz`；曲线摘要：`runs/hestia-cuda-curve-analysis-20260912-001/summary.json`。
- 最新返回：`runs/hestia-image-value-component-recovered-20260912-001/verified/`；manifest SHA256 `cae7302ea1282c6e999420e80f6f0fdf1c1b6c22f92c1185126d808c029774cc`。
- 最新分析：`runs/hestia-image-value-component-analysis-20260912-001/verified/result.json`，SHA256 `aa38ee6ee9445b9c8a372f8d4cf25567b02884b2411b57b4d4e60b477699dcd7`；同目录上层有独立复算及 events。
- 最新远端 identity：`20260912T151222Z-bd007f5402bf-46803787f6a4`；delta SHA256 `302c1d9a07580b57f1bf475e619023ed34d3bf9bc144d12993515eec446ef959`；template SHA256 `090d81a2e46e57377896086582a811a02a3414998b70d918fefc393f80aa3dc8`。
- 最新 run `hestia-image-value-component-20260912-001` 全部完成，不再重跑。收据 `runs/hestia-image-value-component-receiver-20260912-001/receipt.json`；归档 14,213,120 字节。最新 shutdown-request 尚未回传，下次已有授权窗口只读取回；不为补文档单独开 GPU。
- 前一轮 value-route 关机请求已补回，见 `m2-smolvla-hestia-value-route-shutdown-recovery-2026-09-12.json`。不要用它代替最新请求。平台关机观察不等同账单审计，不释放实例。
- 本地四个完整 checkpoint：`artifacts/hestia-native-checkpoints-20260911-001/`。只读数据和权重不进 Git，新 worktree 不会自动拥有被忽略的 `runs/` 和 `artifacts/`；从来源工作区只读定位并核对，不擅自下载或复制全部大目录。

所有本地 ML/数据/数值分析从 WSL Bash 进入固定离线 Docker `sha256:fb3c1bbda42881fac9b7725d3acb436119934f0c60af29747339d5951c3039da`。近期分析使用 2 CPU、2 GiB、network none、仓库只读和精确结果目录可写。PowerShell 仅编辑、Git、静态/hash 检查，不调用 Windows Python。来源工作区为脏功能分支 `codex/smolvla-local-generalization-20260911`，HEAD `bd007f5402bf`，没有提交/推送授权；保留所有修改和历史证据。

用户已授权沿结果继续排查及 SSH，无需再次询问常规可恢复操作。新一轮仍先登记单轴、代码/config/cache 身份、预算、阳性对照、停止条件和受保护回传/关机流程。不能改旧实验脚本后在旧 identity 下重跑，也不能降低 Auto Review、实例保护、隐集隔离或闭环门禁。今晚不启动额外计算。

## 本地 Chrome 开机与 SSH 接续

用户追加明确要求记住本地 Chrome 操作和 SSH 端口 20497，并授权后续自行开机，不再让用户代开。目标 SSH 为 `ssh -p 20497 root@connect.cqa1.seetacloud.com`，目标 AutoDL 实例编号 `44db45aec7-cd880eb6`，重庆 A 区 / 011 机、RTX 4090D 单卡。端口不是实例编号；必须在当前页面重新匹配两者，不能只凭卡型或列表次序操作。

使用 `chrome:control-chrome` skill 的浏览器客户端和本地 Chrome 已登录会话。上轮普通内置浏览器未登录，不能因此让用户再登录或代开。初始化 skill 规定的 browser runtime，选择 `agent.browsers.get("chrome")`，通过 `chrome.user.openTabs()` 找现有 AutoDL 控制台，再 `chrome.user.claimTab(tab.id)`。旧 tab id `525000905` 仅供定位参考，不能假定下一会话仍有效。所有操作走 browser 工具，不用 shell/CDP/cookie 或直接私有 HTTP 请求。

先读取新鲜 DOM，将行精确匹配 `44db45aec7-cd880eb6` 并核对 20497；在该行点击“开机”，读取确认对话框后确认，再核实目标运行状态。旧页面显示的价格和容量可能变化，只把它们当历史。不得点其他两台实例、充值、租新实例、升级资源或释放。GPU 仅在注册诊断确实需要时开启；本地数组分析先行。SSH 认证使用既有密钥，不读取或输出私钥，不把凭证写入命令、日志或文档。按已授权的受保护流程完成保存、回传校验与关机，并观察平台状态；不能把 SSH 断连当作关机证明。
