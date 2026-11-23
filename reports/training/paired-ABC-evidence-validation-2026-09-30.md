# Paired 003 文件审计工具验证

日期：2026-09-30。验证范围仅为标准库单测、静态配置与已回传文件复算；无训练、模型加载/推理、仿真、网络、付费计算。

环境：WSL 启动 Linux Docker；既有镜像 `sha256:4f91ff8e38a2ac6e3e0a307f335f82764c58abfedf15cebf5d67e17fe16b9fc5`（Python 3.13）。`--pull=never --network none --memory 512m --memory-swap 512m --cpus 1 --pids-limit 64 --read-only`，临时目录 64 MiB，旧仓库/实验输入只读挂载，输出目录单独可写。

结果：

- 16 项 unittest 全部通过，0.264 秒。覆盖缺失物理证据不推断、float32 边界、首次事件、reset 安全停止、动作篡改/步号缺口/状态不连续/停止后继续、完整清单篡改与逃逸、bank 持续安全恢复链、禁止审计输出污染原目录、无 ML 导入、上下文脚本 import 无副作用、sidecar 完整张量/阈值/身份漂移/create-only/字节上限。
- 实际测量文件审计通过：151/151 SHA，15 条五场景轨迹共 4,187 步，24 条已创建 bank 接续共 200 步。计数、动作链、registration/summary/reload 绑定通过；`evidence_sufficient_for_causal_attribution=false` 保持为 false。
- 实际上下文审计通过：14/14 登记源文件、bank manifest/audit 与所读 branch/labels 元数据；15 个 processor/config 文件；3 臂 5,000 连续 update、20,000 schedule exposure、共同 D 与学习率精确一致。valid/padding 仅核对保存汇总，未假称逐批 mask 复算。

复现入口如下，须在上述 Linux 容器内执行；`/repo` 为仓库只读挂载，`/out` 为新的可写输出目录。镜像 ENTRYPOINT 已是 Python，编排 `docker run` 时不要再重复传入 `python`。

```bash
python -m unittest discover -s /repo/tests -p test_paired_measurement_files_stdlib.py -v
python /repo/scripts/audit_paired_measurement_files.py \
  --root /repo/.work/paired-ABC-measurement-20260928-003-received-20260930-001/files \
  --manifest /repo/.work/paired-ABC-measurement-20260928-003-received-20260930-001/remote.sha256 \
  --output /out/measurement-audit-new.json
python /repo/scripts/audit_paired_measurement_context.py \
  --repo /repo --measurement-audit /out/measurement-audit-new.json \
  --output /out/context-audit-new.json
```

发布到工作分支的是同一次测试的源码字节副本；没有改动旧 runner 或安全 adapter。代码审阅补丁仅包含本任务新增文件与架构文档追加段，不包含用户先前修改。未 commit、push 或创建 PR。

未验证项：真实 processor、CUDA、model forward、physics step、真机、实时 sidecar 集成、observer parity、性能/磁盘吞吐、新数据采集、新训练效果及正式 Gate。字节上限触发时库抛异常并保留不完整文件；未来 caller 的停止与错误收据还需接入验证。草案配置显式 `draft_not_runnable`、`execution_authorized=false`，不是可直接启动的新实验。
