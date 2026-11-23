# 本轮工作看门狗收紧

用户要求在保留少量缓冲的前提下收紧看门狗。已为正在运行的 `paired-native-ABC-formal-20260927-001` 增加独立工作截止守护，从原始启动时刻计算总期限7200秒，不重新延长两小时。

- 生效截止：北京时间2026-09-28 01:46:05，Unix deadline `1790531165.9964685`。
- 真实armed握手成功，PID21000/start_ticks4236645389，独立session/group21000。
- 原始run注册、训练配置、各阶段预算和原看门狗不改写；原宽松守护继续作为后备，新守护施加更早期限。
- 原worker退出时提前收尾；到期先向匹配PID/start_ticks/session/group的原worker发送SIGTERM，给30秒正常退出窗口；仍存活则用原已绑定清理函数结束所属进程组并清理当前stage。
- 保留已保存checkpoint与退出证据，不将超时视为实验完成。`shutdown_authorized=false`，不执行平台关机。
- 真实标准库进程测试通过：短截止结束被登记的测试worker，不影响无关进程；1 passed / 0.46秒。没有中断或重启训练。

入口：`scripts/tighten_round_deadline.py`。
新source workspace：`20260927T155443Z-66a5bbcdbdcd-2256600de551`，composite SHA `2256600de55158dcfee4e73c80565a1a2deea940d6778399cee586eda03f7f9e`。
本地部署/测试/握手证据：`.work/tightened-deadline-20260927-001-transport/`。
远端收据位于该run的 `deadline-tightening-001/`。
