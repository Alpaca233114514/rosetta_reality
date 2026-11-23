# ALOHA train隔离执行结果

本轮真实运行身份：`aloha-train-isolation-20260927-004`。用户明确允许隔离预处理内部解码混合页、仅导出train。后续用户要求直接SSH修改，视频后续程序在远端新目录创建执行，本地仅接收元数据和本收尾记录。

主任务已完成：29项原回归、5项真实合成Parquet测试通过；human原train40共20,000帧，scripted保守白名单40episode共16,000帧。独立进程核验导出SHA、白名单、完整episode、连续frame/50Hz时间和有限14维state/action。50步chunk有效监督slots分别951,000和751,000。底层可能解码protected页，导出与独立审计观察到的protected行均为0；不声称hidden从未被解码。

主任务耗时2.533秒。11个封存元数据/日志文件共24,030 bytes已回传并逐文件SHA与大小核验。样本只留在远端新namespace的`isolation/train-only`，没有回传本地。

后续视频实验在同一guard有效期间通过SSH新建并执行，远端stdout报告：36,000个train packet时间戳齐全且无重复；每来源train episodes 49/4的开始/中间/末尾，共12帧，解码尺寸与时间匹配；耗时0.738秒。没有导出像素。seek可能内部解码protected参考帧，不向后续分析输出。仅为抽样像素验证，不是完整视频解码或成功标签验证。结果留在远端`video-qualification-001/result.json`；看门狗提前关机后回传连接关闭，该后续原始结果及代码尚未独立回传校验，当前只保留会话stdout观察，不计入主任务11文件。

workspace为`workspaces/20260927T084016Z-66a5bbcdbdcd-a188785d7f32`；composite SHA为`a188785d7f32ae42ed75498a7cf36237df14dc792f7eae9f05b92f0b018166b5`；delta archive SHA为`731f362a10755bfe940d31b0279035140e828aff450e53975a33e84248b4b31b`。16项上传文件SHA一致才启动。worker1438，独立guard1437，工作截止16:45:20、guard硬截止16:47:20、完成回传宽限240秒、平台独立截止16:50。16:45附近重新读取平台时目标实例已关机；未取消timer或释放实例。guard最终shutdown请求本轮未回传，因此不声称精确停机秒数或单独平台因果证明。

## 未解决的科学前置条件

读取阻塞已解决。成功/专家纠正/恢复事件、scene/parent保护和合格S/R matched cohort仍无证据。human仍是原D，不可计新增N；`next.done`不等于成功。单纯修reader不能产生缺失的新增恢复示范。

base原Gate3报告/plan/artifact manifest在本轮与已封存SHA再次核对一致。报告为9次关节限位累计、1次跨臂夹爪接触，两个安全标准失败；代码按实际物理状态和接触计数判定。未找到可直接修正的判定缺陷，未将真实失败改为passed。base Gate4继续not_measured。

本轮没有模型forward、optimizer、三臂训练、仿真或新Gate结果；M2不变。下一阶段需要真实同域新增恢复数据或新的合格采集来源，而不是将缺标签/下载完整性转换成训练准入。已有teacher候选未通过独立门槛，不能自动作为标签生成器。用户已被询问是否有尚未登记的示范路径/来源。
