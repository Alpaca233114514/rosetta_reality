# Hestia scene-kv-mean 固定后处理

身份 `hestia-scene-kv-mean-analysis-20260913-001`。在本轮 mean 条件的效果尚未取回或查看前，
固定本后处理；不增加科学干预，不调参、不访问 hidden。

只接受新可执行模板绑定的完整十条件与封存 manifest。要求 worker 无错误、0 optimizer、
各条件参数不变、180 forwards、target/noise 一致，两个原生端点和四恢复对照完整数组精确
复现旧 CUDA。未完成或失败时只保留失败证据，不生成完整科学结论。

只读已核验来源的早期差距 helpers，逐项 SHA 后再 import。全量报告两 checkpoint、
train40/dev5、标准/归一化、完整/首动作、左右及合并关节/夹爪，逐噪声与场景。
保留训练标签均值/中位数、训练预测均值模板和 MSE 三项分解。
另以成对移动平方和残差对齐恒等式验证 intervention-minus-native MSE 差。

source 和本工作区只读；固定离线 Linux Docker、2 CPU、2 GiB、300 秒，只有新分析目录
可写。先通过成对差符号和噪声不可混合的合成反例。仅输出新 JSON，不运行本地模型。
任何 SHA、shape、finite、历史复现或恒等式失败立即停止。容差保持 1e-12。
主要终点和结果解释边界沿用 `scene-kv-mean-plan-2026-09-13.md`，不能借后处理新增实验轴。
