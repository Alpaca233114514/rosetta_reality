# Rosetta Reality 下一步实验报告与研究方向（数据集与泛化能力视角）

日期：2026-11-22  
状态：只读远端核查 + 本地证据综合分析 + 下一阶段研究规划（**未启动新训练、下载、数据修改、模型推理或删除操作；所有建议方向均为未授权提案**）。  
当前项目主结论：**M2 仍未完成**；截至目前全部已完成正式 Gate 4 与 Paired 闭环评测仍为 `0/5`，无新 checkpoint 选择或闭环通过证据。

---

## 1. 远端服务器核查与关机收尾记录

本次于 2026-11-22 18:45–18:51（CST）通过 SSH 接入已指定 AutoDL 实例（端口 `44126`，容器名 `autodl-container-179442a4e1-7081e39b`）执行只读盘点，并在核查完成后立即触发容器关机，避免本地分析占用计费时长。

| 检查项 | 实测结果 | 备注与风险 |
|---|---|---|
| 硬件与计算模式 | 无卡模式（`/dev/nvidia*` 不存在），128 vCPU，503 GiB 主机内存 | 未挂载 GPU，无法直接跑 CUDA 训练/推理 |
| 运行进程与会话 | 无 `tmux` 会话，无 `python`/`train`/`eval`/看门狗任务 | 仅系统基础守护进程在运行 |
| 磁盘占用 | `/` 已用 26G/30G（**86%，仅剩 4.4G**）；`/root/autodl-tmp` 已用 38G/50G（**75%，剩 13G**） | 空间偏紧，后续保存新 checkpoint 或扩数据集前需先受控清理 |
| 主要目录占用 | `checkpoints` 23G、`artifacts` 6.8G、`workspaces` 3.3G（167 个快照）、`control` 1.6G、`envs` 1.5G、`models` 915M、`tasks` 437M、`runs` 255M、`dev` 241M、`data` 88M | 23G 主要为 2026-09-10 至 09-14 历史 smoke/full checkpoint |
| 缓存数据集 | 仅 1 份：`lerobot/aloha_sim_insertion_human`（rev `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`，v3.0，50 episodes / 25,000 frames，50 Hz，单目 `observation.images.top` 480×640，14 维 state/action） | 机上无腕部相机视角、无其他 ALOHA 任务、无新增多场景数据集 |
| 缓存模型 | `lerobot/smolvla_base`（rev `c83c3163b8ca9b7e67c509fffd9121e66cb96205`）与 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 依赖环境位于 `envs/smolvla-cuda-001` |
| 最新远端活动 | 停留在 2026-10-11/12 的 `dataset-variable-20261011-003`（workspace `20261011T025059Z-517b24bcb817-5b290aa746ac`） | 产物已于 2026-10-12 回传至本地 `.work/autodl-yesterday-complete-20261012-001-received/` 并完成清理，远端无未回传新结果或残留 `shutdown-request.json` |
| 关机执行与核验边界 | 已核验 `/usr/bin/shutdown` SHA-256 精确匹配 `0358e83eeeaf542aa98f64ba9e339c91df46f1e025892d52dba159f4fb1cf027`、确认无 Trash 及活跃 worker 后执行 `sync` + `/usr/bin/shutdown`，SSH 主动断开；二次探活返回 `Connection closed ... exit 255` | **按 `AGENTS.md` 合同披露**：当前会话无已认证 AutoDL 网页控制台工具，未刷新平台页面核验“已关机”状态；关机收尾仍待你在平台控制台一眼确认，本次未释放实例或删除任何远端数据 |

---

## 2. 既有实验全景复盘：离线降 93% 却全部 0/5 的证据链

综合 2026-08-12 至 2026-10-14 全部已验证报告（`reports/training/` 与 `.work/dataset-window-analysis-20261014-001/`），过去两轮系统性排查已经排除了大量常见工程与单一超参嫌疑，但也清晰暴露出**“离线拟合指标改善”与“闭环泛化成功”之间的结构性断层**：

| 实验线 / 身份 | 变更单轴与训练预算 | 离线 / 诊断关键结果 | 闭环结果（Gate 4 / 诊断） | 已排除 / 留存结论 |
|---|---|---|---|---|
| **Faust** (`b8-002`, 08-12) | 动作边界修复（bounded-sine gripper）+ 2,500 步 / 20k 曝光 | Val MAE `0.1705 → 0.0447`（降 73.81%），reload 差 0.0 | Gate 3 通过；Gate 4 **0/5**（seed 1003 max reward 2） | 排除基础动作非法/序列化错误 |
| **Zen** (`411`/`422`, 08-27) | `first_action_only` 时序加权 vs uniform；b64 / 316 步 / 20k 曝光 | First-action MAE `0.2905 → 0.0216 / 0.0222`（降 >92%） | Gate 3 通过；双臂 Gate 4 **0/5**（422 甚至 0 碰撞 0 越限但 reward 全 0） | 否证“仅加大首动作 loss 权重即可修闭环” |
| **vcdropout / vfunfreeze** (`433`/`444`, 08-29 / 09-06) | 状态 dropout / 视觉前端解冻；b32 / 632 步 / 20k 曝光 | `vfunfreeze` 首动作 MAE **0.019603**（历史最低，降 93.25%），首个通过 frame-0 视觉对齐门 | Gate 3 通过；Gate 4 **0/5**（19 越限 / 19 碰撞） | 否证“视觉前端冻结/看不到图是唯一根因” |
| **Canonical fullframes** (`001` / Gate 471 / Traced 005, 09-14–16) | 完整 20,000 帧遍历 + Canonical RGB 统一缩放；b4 / 5,000 步 | 20k 唯一帧精确曝光 1 次；原生/导出/适配器 chunk 逐值相等 | Gate 3 通过；Gate 4 **0/5**（004: 19 越限/38 碰撞；005: 27 越限/60 碰撞，seed 1002 仅 3 步 reward 2） | 排除 RGB 缩放偏差与帧覆盖不全作为唯一原因 |
| **Hestia 机理排查** (09-11–13) | 640 vs 1280 步 K/V 互逆、逐层 V、路由、图像 V 分量、Q 回放 | 640 步已存在 **77.42%** 最终左关节泛化差距；图像 V 共同均值偏移解释 640→1280 后期退化的 35.9%；640 步 K-mean 可将左关节 MAE 从 0.0585 压至 0.0510 | 0 optimizer 机理排查；全部条件均未同时击败 train-only 常数基线 | 定位了后期图像 V 漂移与早期 K 场景匹配弱，但尚未定位唯一根因 |
| **Iris 002** (`451`/`452`, 09-13–14) | 图像 K 场景正则化；1,280 步 | 左关节开发 MAE 仅降 1.04%，仍差于 train-only 常数基线 | Gate 3 通过；双臂 Gate 4 **0/5**（0 安全违规但 reward 全 0） | 单纯 K 正则化不足以建立泛化与闭环成功 |
| **Paired ABC 003 + 物理定位** (09-28–10-04) | A(20k D) vs B(15k D + 5k R) vs C(15k D + 5k S)；5,000 步 | R/S 辅助仅 1 个 parent（2436），偏离后样本曝光仅占 6.05%；2445 首步 B/C 碰桌由 `right_shoulder`（6.3× 位移）双向因果触发；A 为多臂耦合诱发反侧手指越限 | 探索评测均 **0/5**（4/15 跑满 500 步 reward=0，11/15 安全早停；bank 24/24 早停） | 排除单维全局夹爪偏置；暴露恢复覆盖极度不足与首步物理动力学耦合 |
| **Dataset-variable 003** (`n08/n20/n40`, 10-11–14) | 固定 2,500 步 / 10,000 曝光（b4），训练轨迹数 8 → 20 → 40（每条 250 帧） | Dev 四项平均 MAE 降 **15.21%–21.18%**；但 frame 0 四项全面恶化（-6.5% 至 -22.8%），ep45 四项全面恶化，frame 125 夹爪误差占 60.7% 且 96.2% 为均值失配 | `checkpoint_saved=false`，未测闭环 | **关键正面线索**：扩大轨迹覆盖显著改善平均泛化，但单遍预算引发局部欠拟合且长尾/接触窗仍未解决 |

---

## 3. 核心瓶颈诊断：从“数据集”与“泛化能力”切入

### 3.1 数据集层面的四个结构性瓶颈

1. **轨迹数量与初始状态覆盖天花板过低（40 条远未饱和）**
   - 当前整个项目长期锁死在 `lerobot/aloha_sim_insertion_human` 的 **40 条训练轨迹**（全集 50 条 = 40 train + 5 dev + 5 hidden）。
   - `dataset-variable-20261011-003`（`reports/training/dataset-variable-analysis-2026-10-14.md`）给出了最直接的实证：从 8 条扩到 40 条，开发集首动作关节、首动作夹爪、完整 chunk 关节、完整 chunk 夹爪平均 MAE 分别下降 **15.71% / 17.59% / 21.18% / 15.21%**，且三个推理噪声种子一致改善。
   - 但 40 条依然覆盖不足：5 条 dev 中，**episode 45 在 8→40 后四项全面退步（-21.82% / -24.49% / -9.33% / -18.34%）**，其中 frame 125 左夹爪首动作误差高达 `-0.593` 至 `-0.633`；且去掉单条 episode 7 后，8→40 的 chunk 夹爪平均收益直接从 `+15.21%` 翻为恶化 `-4.78%`。这说明 40 条数据在双臂 peg/socket 相对位姿空间里仍是极度稀疏的采样。

2. **训练预算把“覆盖扩展”与“样本重复/收敛充分度”强耦合，且量级远低于 SmolVLA 常规微调区间**
   - 在 `dataset-variable-20261011-003` 中，总更新固定 2,500 步（10,000 曝光），8→20→40 把单输入重复次数从 **5 次压到 1 次**。结果虽然中后段泛化变好，但 **frame 0 起步动作四项全面恶化**（首动作关节 `-14.52%`，首动作夹爪 `-22.83%`；`.work/dataset-window-analysis-20261014-001/uncertainty-v2.json` 显示 frame 0 右夹爪在 15 个配对条件中 **13/15 恶化**，且 58.8%/81.4% 的 MSE 来自均值偏差而非随机噪声），同时 train-probe 拟合误差单调升高（首动作关节 `0.0271 → 0.0360`）。
   - 对照 SmolVLA 原论文（`arXiv:2506.01844` §4.3）：在 LIBERO / Meta-World 仿真微调冻结 VLM 时，默认配置为 **batch 64 × 100,000 steps**（即使社区小规模微调也通常在 20k–50k steps 级别）。而 Rosetta 截至目前的正式炉大多停留在 **316–5,000 步（仅 0.5–1 个 epoch，10k–20k 总曝光）**。在 40 条数据、单遍曝光的 regime 下，模型既没见过足够多样的场景，又无法在扩大覆盖后充分拟合关键相位动作。

3. **观测模态信息不完备（单目顶视角遮挡 + 缺历史状态）**
   - 远端数据集清单核实：`cc571a3c...` 仅包含单路顶视角 `observation.images.top`（480×640），**没有左右腕部相机**。
   - `AlohaInsertion-v0` 是典型的高精度双臂接触操作（peg 插入 socket 容差在毫米级）。在抓持建立（frame 125 附近）与对准插入阶段，双臂手腕和夹爪会对 peg/socket 形成强自遮挡；而源 geometric teacher 享有特权 EEF/物体 6D 位姿、接触事件与阶段历史，学生每步仅靠单帧顶视图 + 14 维逻辑关节状态（且不含夹爪反侧手指物理位置）进行单帧无状态重规划（`reset policy`），存在显著的**部分可观测性（POMDP）与单帧动作歧义**。

4. **缺乏多场景 On-Policy 恢复数据，且 5 条 dev 已被反复诊断耗尽独立性**
   - Paired B/C 的 R/S 恢复数据实际仅来自 **1 个 parent（2436）**，偏离后样本仅占全部曝光的 **6.05%**（`N_exploratory=1, N_strict=0`），这远不足以检验 RaC / DAgger 所要求的“在策略真实失败邻域覆盖多场景恢复”。
   - 同时，dev `[22, 13, 7, 33, 45]` 已被连续诊断两个月，单条轨迹（如 ep7、ep45）即可翻转聚合指标符号，亟需在**不解封 5 条 hidden test** 的前提下建立更大样本量的独立仿真验证池。

---

### 3.2 泛化能力与闭环转化层面的三个核心断层

1. **接触关键窗（frame 125）的系统性均值失配，而非随机噪声**
   - 根据 `reports/training/dataset-variable-analysis-2026-10-14.md` 与 `.work/dataset-window-analysis-20261014-001/uncertainty-v2.json`：
     - 40 条组在 frame 125 的夹爪首动作 MAE 仍为 **0.20046**，独占四个评估时间点总误差的 **60.7%**；
     - 在 3 个推理噪声种子的方差分解中，dev frame 125 左夹爪的 MSE（首动作 `0.12965` / chunk `0.08306`）里，高达 **96.20%（首动作）与 91.89%（chunk）** 来自平均预测误差平方（bias = `-0.211`，显著滞后/过闭），3-seed 噪声方差仅占 3.8% / 8.1%。
     - 40 条模型在开发集上的 chunk 夹爪 MAE（`0.06733`）仍比仅用 8 条训练轨迹构建的帧对齐均值基线（`0.05075`）差 **32.7%**。说明模型在“何时开爪、何时闭合”的视觉-状态相位泛化上仍存在本质缺陷。

2. **早期视觉条件化建立不足 + 后期图像 V 漂移的双重退化**
   - Hestia 系列（`reports/training/m2-smolvla-hestia-root-evidence-result-2026-09-13.md`、`scene-kv-result`、`image-value-component-result`）表明：640 步时就已形成最终左关节泛化差距的 **77.42%**；在 640 步用训练集均值替换图像 K 可将左关节 MAE 从 `0.0585` 改善至 `0.0510`（增强正确场景协方差），说明早期跨注意力 Query-Key 场景匹配本身学偏了；到 1280 步时，真实图像 64-token 上的共同 V 偏移又叠加了新的双向有害退化（占后期退化 35.9%）。换言之，少量轨迹上的微调容易让 action expert 对固定背景/状态产生过度耦合的 K/V 映射，换到未见物体位姿时泛化急剧衰减。

3. **50/1 闭环执行节奏与物理动力学放大效应尚未量化**
   - 现有全部闭环评测固定采用 **50 槽预测、每步只执行 slot 0 并重置策略（50/1）**，每步重新采样独立高斯噪声；但 SmolVLA 原论文 §4.7 专门对比过 `n_action_steps ∈ {1, 10, 30, 50}`，在 50 Hz 位置控制下高频随机重规划极易诱发高频命令跳变。
   - `reports/training/paired-physical-command-localization-2026-10-04.md` 已给出实证：在 2445 恢复初态上，B/C 仅因 `right_shoulder` 单维目标位移达到 teacher 的 **6.3 倍**（`-0.089` vs `-0.185 rad`）就在单个 20 ms 步内撞桌；A 则是多臂关节大跳变通过动力学耦合把未观测反侧夹爪手指顶出 `-0.057` 物理下限。另外 2447 C 在双爪持续命令闭合时，因臂部伴随运动直接丢掉右侧接触。这些都表明：即便单步离线 MAE 看似很小（`0.02–0.04 rad`），**闭环高频重规划 + 动力学放大 + 缺少恢复策略** 会迅速把微小误差放大为抓偏、丢接触或安全早停。

---

## 4. 下一步实验方向与分阶段路线图（待立项授权）

严格遵循 `AGENTS.md` 单变量轴、先低成本诊断再重训、完整 provenance 与闭环门禁原则，建议按 **P0 → P1 → P2 → P3** 推进，每一步均设清晰停止/转向条件：

### P0：零训练、极低算力的闭环执行与边界诊断（优先度最高，预计单次短 GPU 窗口完成）

在启动任何新一轮训练前，先用已保存的现成 checkpoint（`optimizer_updates=0`）回答两个“花小钱定方向”的关键问题：

- **方向 P0-A：执行已就绪的 2445 最小推理边界 pilot**
  - **入口**：已绑定 13 项合成收据的 `configs/diagnostics/paired-boundary-pilot-20261004-002.json`（`scripts/paired_boundary_pilot.py`，最多 5 次 forward / 6 个物理步，0 optimizer）。
  - **目的**：先验 exact restore、A1/A2 同噪声重复性、observer parity 与真实生效 processor stats/shape/dtype，把“工具链或推理边界残留不一致”与“纯策略动作生成误差”彻底切开。
  - **转向条件**：若发现 restore/processor/parity 异常，先修工具链再谈数据与训练；若 parity 完整通过，则彻底锁定问题在**策略泛化与闭环执行侧**。

- **方向 P0-B：零训练单轴扫描执行步长 `n_action_steps` 与推理噪声（复用现成 checkpoint）**
  - **设计**：固定已验证的 step-5000 checkpoint（如 `canonical-fullframes-20260914-001` 或 `vfunfreeze-003`），0 optimizer，仅改变闭环执行步长 `n_action_steps ∈ {1, 5, 10, 25, 50}`，并额外加一组 `zero-noise`（固定初始零噪声确定性 ODE 流）对照，评测统一记录细粒度阶段里程碑（接近距离、首次物体接触、双侧抓取保持步数、peg-socket 接触、安全事件明细），不再只看二值 0/5 成功率。
  - **科学价值**：直接检验“50/1 每步重采样噪声导致的命令抖动/跳变”是否压垮了本已学到的局部轨迹平滑性；若增大 `n_action_steps` 或降低噪声后接触率/reward 明显跃升，则说明后续应优先固定合理 execution horizon（并可单独评估 ACT 风格 temporal ensembling）；若全部 `n_action_steps` 仍无法建立稳定抓取，则彻底确认瓶颈在**数据与策略泛化本身**。

---

### P1：数据集扩展与“覆盖-收敛”解耦（提升泛化能力的核心主线）

基于 `dataset-variable-20261011-003` 的正面信号（8→40 平均误差降 15%–21%），下一轮真正值得投入 GPU 算力的主轴应放在**数据集规模、多样性与信息完备性**上，而非继续在 40 条数据上微调 loss 或正则化：

- **方向 P1-A：建立可验证的闭环“正参考基准”（基准难度与收敛下界标定）**
  - **问题**：目前在 40 条 `aloha_sim_insertion_human` 单目顶视角上连续 10+ 轮全 0/5，缺乏“SmolVLA 在本仓库训练/评估链路下确实能打出 >0% 闭环成功率”的阳性基准。
  - **建议**：在保持同一训练与仿真代码栈不变的前提下，先做一项**阳性对照标定**（二选一）：
    1. 在同一 `AlohaInsertion-v0` 任务上，使用更大规模的仿真轨迹（如将训练集从 40 条扩充至 **200–500 条**，先在容易采集的 scripted/多场景合格 teacher 轨迹上验证）；或
    2. 在更低容差门槛的官方 LeRobot ALOHA 仿真基准 `lerobot/aloha_sim_transfer_cube_human`（同 14 维双臂契约，仅换任务）上跑一次标准多 epoch 微调，确认本仓库 SmolVLA 闭环链路本身具备打出正成功率的完整能力。
  - **验收意义**：一旦阳性对照跑出明确闭环成功率（例如 ≥20%–40%），就能立刻量化“当前 insertion 0/5 到底差在数据规模、训练步数还是任务单目遮挡”。

- **方向 P1-B：解耦“轨迹覆盖数”与“重复次数 / 训练步数”（修复 frame 0 退步）**
  - **设计**：解决 `dataset-variable-20261011-003` 的两个遗留缺陷（单遍重复=1 导致 frame 0 退步 + 未保存 checkpoint 无法做闭环评测）：
    - 独立对比两个正交维度：
      1. **固定 40 条，扩训练预算（多 epoch 收敛轴）**：在统一 batch 下将更新步数推进至 **10k / 25k / 50k steps**（对照 SmolVLA 论文微调量级），同步监控 train/dev 的 frame 0、frame 125 夹爪、ep45 长尾与闭环阶段指标，判断“欠训练”还是“40 条过拟合平台”先到来；
      2. **固定单样本充分重复（等 epoch / 等每帧曝光），扩轨迹覆盖（数据规模轴）**：比较 `40 条 vs 100 条 vs 200+ 条` 独立初始位姿的插入演示（按 scene/seed 严格隔离 train/dev，保留至少 2 个独立训练 seed 校验方差，且必须保存可独立 reload 的 checkpoint 以进入 Gate 3/4）。
  - **强制指标看板**：每轮除总体 MAE 外，必须单列 **frame 0 起步误差、frame 125 夹爪误差、最差 20% 场景（含 ep45）误差**，以及闭环 **接近/接触/抓持保持/插入** 四阶段通过率，杜绝总体均值掩盖局部退步。

- **方向 P1-C：引入多视角观测（顶视 + 左/右腕部相机）解决接触遮挡**
  - **动机**：SmolVLA 原生架构即支持多相机视觉流（前缀已有 3×64 图像槽，目前其中 **128 个空相机 token 处于浪费占位状态**，`AGENTS.md` 与 Hestia 前缀审查均已证实）。当前仅用顶视角单目完成毫米级双臂插孔，等于人为丢弃了 SmolVLA 原生多视角容量与手眼近距几何信息。
  - **设计**：作为独立单轴，在 MuJoCo 仿真数据采集与推理 adapter 中接入双腕相机（填充原本空置的两路相机槽），对比“单顶视”vs“顶视 + 双腕”在 **frame 125 夹爪相位误差、2447 抓取保持稳定性与闭环接触/插入成功率** 上的差异。

---

### P2：面向闭环分布漂移的恢复数据与泛化增强（在 P0/P1 明确基线后启动）

- **方向 P2-A：多 Parent、多阶段 On-Policy 恢复数据（真正落实 RaC / DAgger）**
  - **修正 Paired ABC 003 的实验局限**：不再只用 1 个 parent（2436）人工偏离，而是从 P1 训练出的基础策略（已能稳定接近甚至偶发接触的 checkpoint）的实际闭环 rollout 中，抽取 **≥15–30 个不同初始场景（独立 parent）** 的失败前邻域状态（覆盖“接近偏斜”“抓取滑脱”“对准偏心”三类关键窗口），用安全验证通过的 teacher 生成恢复链。
  - **协议约束**：严格按 parent 划分 train/dev；透明统计前缀/偏离/恢复/完成四段曝光与有效 target 槽数；单独测试混合比例（如 IWR 风格 1:1 重加权）作为独立实验轴，不与数据量扩展或 horizon 修改混在一起。

- **方向 P2-B：轻度视觉与状态数据增强（对抗背景记忆与 K/V 捷径）**
  - **动机**：结合 Hestia 发现（早期图像 K 场景匹配弱、后期图像 V 共同均值漂移），当前训练配置 `augmentation=false`、`state_dropout=false`（见 `.work/dataset-window-analysis-20261014-001/datasets-v2.json`）。在数据量扩充到 100+ 轨迹且训练进入多 epoch 后，可单轴引入轻度图像光度/随机微裁切增强（保持像素几何语义不破坏空间定位）或极小幅本体状态高斯扰动（例如 `σ = 0.005–0.01 rad`，远小于此前 0/5 的大比例 dropout），抑制状态捷径并提升对未见物体位姿与轻微闭环漂移的鲁棒性。

- **方向 P2-C：构建防过拟合的新仿真泛化评估基准**
  - 保留原 5 条 dev `[22,13,7,33,45]` 作为历史纵向对照锚点，**绝不解封 5 条 hidden test `[31,6,1,24,5]`**；
  - 新增一套独立的 **20–50 个固定新 seed 仿真评测集**（区分分布内初始位姿 vs 外推/扰动初始位姿），同时输出：任务成功率、四阶段里程碑（接近/接触/抓持≥5步/对准）、安全违规分型（明确记录越限关节名、反侧手指裕度与碰桌接触对），避免 5 条小样本造成的高方差误判。

---

### P3：远端存储治理与工程前置（开展 P1 训练前必做）

本次 SSH 核查显示远端磁盘已接近安全红线（`/` 剩 4.4G，`/root/autodl-tmp` 剩 13G），如直接启动多 checkpoint 训练或扩增仿真视频数据集极易再次触发 `vfunfreeze` 曾发生过的写盘打满故障：

1. **先盘点核验、再按授权清理远端历史冗余**：远端 `/root/autodl-tmp/rosetta/checkpoints` 占 23G（13 个 9 月 smoke/full 目录）、`artifacts` 占 6.8G、`workspaces` 占 3.3G（167 个历史快照）。对照本地及云端已完成 SHA 校验的备份清单，仅保留当前仍需复用的基线 checkpoint（如 `canonical-fullframes` 与 `ABC step5000`），将已归档历史目录清理，预计可安全腾出 **20–28G** 持久数据盘空间（清理前须逐路径列出并获明确授权）。
2. **固定“新实验必须保存可 reload checkpoint”规范**：后续涉及泛化改进的对比实验，避免再次出现 `dataset-variable-20261011-003` 中 `checkpoint_saved=false` 导致有离线提升却无法测闭环的遗憾。

---

## 5. 推荐执行优先级与决策入口

| 优先级 | 建议动作 | 算力/时间估算（草案，未报价） | 核心回答的问题 | 启动所需授权 |
|---|---|---|---|---|
| **第 1 步（P0）** | 跑 `paired-boundary-pilot-20261004-002`（5 forward）+ 现成权重 `n_action_steps ∈ {1,5,10,25,50}` / 零噪声闭环扫描 | 短 GPU 窗口（预计 ≤1 小时，0 optimizer） | 彻底排除推理/processor 边界问题，并测清 50/1 高频重规划对 0/5 的贡献 | 授权开单卡 GPU + 登记新 run ID |
| **第 2 步（P3）** | 按备份核验清单清理远端 9 月历史冗余 checkpoint/workspace | 无卡 CPU 窗口（约 15 分钟） | 释放 20–28G `/root/autodl-tmp` 空间，避免后续写盘爆满 | 逐路径清单确认授权 |
| **第 3 步（P1-A/B）** | 阳性闭环对照标定 + 解耦“轨迹覆盖（40→200+）vs 多 epoch 步数（10k–50k）”实验（保存 checkpoint 并测新 20-seed 阶段里程碑） | 中等 GPU 窗口（分阶段注册） | 验证数据规模与充分收敛能否首次打破 0/5，并修复 frame 0 与 ep45 长尾 | 授权新数据生成/下载 + 正式训练计划 |
| **第 4 步（P1-C / P2-A）** | 引入双腕相机（填满 SmolVLA 3 路图像槽）或采集 ≥15 parent on-policy 恢复数据 | 中等 GPU 窗口（单轴依次进行） | 攻克 frame 125 接触遮挡与闭环分布漂移，冲击真实 Gate 4 达标 | 基于前序实测结果单独冻结单轴计划 |

---

## 6. 证据来源索引（全部为仓库相对路径）

- 架构与门禁总览：`AGENTS.md`、`docs/m2-smolvla-architecture.md`
- 历史核心负结果审计：`reports/training/m2-smolvla-faust-trainer-optimizer-audit-2026-08-12.md`、`reports/training/m2-smolvla-zen-formal-audit-2026-08-27.md`、`reports/training/m2-smolvla-vfunfreeze-gate34-closure-2026-09-06.md`、`reports/training/m2-smolvla-canonical-fullframes-gate34-result-2026-09-15.md`
- 视觉泛化与 K/V 机理排查：`reports/training/m2-smolvla-hestia-next-session-handoff-2026-09-12.md`、`reports/training/m2-smolvla-hestia-image-value-component-result-2026-09-12.md`
- 闭环失败分型与物理定位：`reports/training/paired-ABC-evidence-audit-and-next-plan-2026-09-30.md`、`reports/training/vla-objective-execution-investigation-2026-10-03.md`、`reports/training/paired-failure-localization-2026-10-03.md`、`reports/training/paired-physical-command-localization-2026-10-04.md`
- 数据集轨迹覆盖与窗口分解：`reports/training/dataset-variable-analysis-2026-10-14.md`、`.work/dataset-window-analysis-20261014-001/datasets-v2.json`、`.work/dataset-window-analysis-20261014-001/window-analysis-v2.json`、`.work/dataset-window-analysis-20261014-001/uncertainty-v2.json`
- 文献参考：SmolVLA (`arXiv:2506.01844` §4.3, §4.7)、ACT (`arXiv:2304.13705`)、RaC (`arXiv:2509.07953`)、IWR (`arXiv:2012.06733`)、DAgger (Ross et al., 2011)
