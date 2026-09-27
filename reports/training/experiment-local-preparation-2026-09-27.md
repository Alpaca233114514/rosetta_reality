# 实验本地资格准备与guard修复（2026-09-27）

身份：`experiment-local-preparation-20260927-001`。用户明确要求在本地推进前一轮提到的修改。本轮已完成工具实现、离线合成验证与文档登记，没有SSH、远端上传、开机、数据下载或真实实验。

## 实现

- `scripts/audit_targeted_aloha.py`：读取固定source manifest及Parquet footer，显式开关控制train-only流式行检查；protected/unknown row group在读取前整组排除。新增行检查不是成功/恢复标签生成器，视频/物理语义/监督有效slots仍需专门证据。
- `configs/diagnostics/aloha_qualification_20260927_001.json`：已用现有最终receipt hash固定ALOHA human source及原train/dev/hidden split。此来源属于D，不能冒充新增S/R；scripted完整receipt仍待封存。
- `src/rosetta_reality/diagnostics/cohort_qualification.py`：hash绑定逐episode合同、rows及labels报告，拒绝unknown、failed-recovery、time-indexed replay、缺teacher gate、合同不一致及protected/重复parent/content碰撞，检查一对一S/R的5%匹配约束。报告内容真实性和完整保护清单仍需独立审查；其输出永远不是训练适配manifest。
- `scripts/autodl_bounded_guard.py`：新版本保留旧guard和失败记录。退出worker正常处理，pid复用/group不匹配不发信号，TERM/KILL间复核身份；cleanup异常保存报告后继续关机安全检查。要求显式关机授权并使用monotonic截止，保持wrapper hash、Trash、GPU和无关进程保护。平台独立定时关机仍为必需兜底。
- `docs/aloha-cohort-qualification.md`与架构入口已同步，原设计、Gate、Action Contract及oracle v1不变。

## 验证和实际范围

既有登记镜像在当前Docker引擎不可见；只读确认原D盘Docker数据文件存在，未改配置、删除或重建旧数据。为了纯标准库检查，用已安装且`dpkg -V`无差异的Debian Python3.13标准库构建无网络FROM-scratch本地镜像，没有pip/apt安装或外部镜像下载。

测试镜像固定digest：`sha256:4f91ff8e38a2ac6e3e0a307f335f82764c58abfedf15cebf5d67e17fe16b9fc5`。
Debian包版本：Python3.13 `3.13.5-2+deb13u3`、libc6 `2.41-12+deb13u3`。
WSL Bash发起Linux Docker，2CPU、512MiB、相同swap上限、无网络、只读源码、128MiB临时盘。

最终 **26项检查通过**，耗时约0.074秒（仅容器内测试计时）。覆盖真实子进程组终止与无关进程存活、已退出proc/stat、pid复用、复杂comm解析、cleanup失败、关机授权/wrapper漂移/create-only证据，以及缺标签、失败恢复、replay、teacher gate、源hash、合同、protected分组/episode/content、路径穿越和配对失配。没有调用真实shutdown。

测试原始输出：`.cache/qualification-tests-20260927-002/{tests.log,result.json}`。
镜像来源与构建证据：`.cache/qualification-stdlib-image-20260927-001/`。
机器读出与源hash：`reports/training/experiment-local-preparation-2026-09-27.json`。
`git diff --check`通过，Bash入口语法检查通过。已有用户修改和旧失败包全部保留，没有commit/push。

## 尚未验证

标准库容器没有PyArrow，也没有真实远端数据。因此真实Parquet读取、统计可信性、视频解码/时间对齐、普通成功/专家恢复事件、scene/parent完整清单与matched cohort均未验证。本地合成检查不等于数据合格、门禁通过或任务成功。

远端guard修复尚未部署和实测；平台关机状态只沿用本聊天此前已独立核验的关机收据，本轮没有重新核验服务器。所有新数据资格工具都保持`training_ready=false`，Exp1现成有效base Gate3绑定仍待补齐，M2仍未完成。

下一次实际执行应先跑Q资格审计：封存scripted完整receipt与保护组清单，审核真实数据/标签/视频，确认可用S/R及匹配N；之后登记sampler、恢复bank和runtime/资源预算，再进入smoke和训练。若共享hidden row group无法安全读取，应保留跳过证据并设计隔离reader，不能降低排除条件。不要在数据比较中顺手改Gate或造专家标签。
