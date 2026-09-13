# Hestia 图像 K/V 场景均值干预：本地实现与启动前验证

已完成新身份 `hestia-scene-kv-mean-20260913-001` 的本地实现、合成验证和版本化传输包。
尚未 SSH、上传、开 GPU 或执行模型；没有新增根因结论。
用户最新要求仅围绕根因推进，只有需要 SSH 时再询问；本次推进至该边界。

## 这次能回答的问题

现有证据表明 640 步左关节的场景响应缺少正确动作对应。新实验在同 checkpoint 内分别
移除真实图像 K 或 V 的跨场景差异，判断哪个位置承载了有害响应。640 是主诊断，1280
为后期对照，不进行 checkpoint 选择。结果不自动推出唯一根因、修复或闭环成功。

科学设计保留在同日期 `scene-kv-mean-plan.{md,json}`；其“尚未实现”字段记录登记时点，
不改写历史。新的可执行入口为 `scene-kv-mean-template-2026-09-13.json`。

## 实现与实际验证

- `scripts/hestia_scene_kv_mean.py`：每 checkpoint 单独封存 train40 图像 K/V 均值。
  保留完整 241 位置 GEMM，只替换真实图像 64 位置。检查 73 个有效位置和完整 mask。
  每行/层/投影原生输入及输出绑定 hash，跨四噪声和十去噪调用须逐值重复。
- 同 checkpoint 的 K/V restore 使用 mean 分支后直接恢复原始输出；不是 BF16 残差加减。
  两原生端点、四恢复对照全部逐值通过后，才能运行四个 mean 条件。
- `scripts/diagnose_hestia_scene_kv_mean.py`：复用按 SHA 核对的原生采集路径，保留
  action/normalization、raw image、noise、参数不变和范围检查，没有跨 checkpoint donor。
- 新 supervisor、streamer、receiver、verifier 保留看门狗、截止时间、源文件哈希、
  白名单回传和匹配收据后的保护关机。没有执行任何关机或平台操作。
- 固定离线 Linux Docker 中环境检查、Ruff 与格式检查通过。
  **41 项新合成/CLI 测试 + 69 项远端入口相关回归测试 = 110 项通过。**
  本地无 CUDA/XPU 设备，没有构造真实模型。CPU 通过不代表原生 CUDA 干预通过。
- 旧模板 85 项输入中 83 项在来源工作区逐项 SHA 匹配；另两项是昨日图像 donor
  对照数组的远端专用别名，本实验不使用，因此新模板明确移除。没有放宽必需输入检查。
- 640/1280 的 20 个 checkpoint 文件已在本轮前序静态核验；本实现保持该身份。

## 源码与传输边界

所有新增文件仅位于 `codex/hestia`；来源工作区只读，未提交、推送或移动分支。
新 worktree 的旧 HEAD 不是 Hestia 实现来源。测试在容器临时工作区复制来源的小型
scripts/tests 并加入新文件，src/configs 只读引用来源；不复制权重、数据或完整 runs。

传输包包含 11 个新文件：5 个 Python 实现、1 个接收 shell、2 个测试和 3 个计划/template。
未向旧 identity 覆盖代码。包内全部常规文件、长度、SHA 与白名单已独立验证：

- 归档：`runs/hestia-scene-kv-mean-preparation-20260913-001/delta.tar`。
- 归档大小：112,640 字节；有效文件内容 101,938 字节。
- SHA256：`0cb431b543ee98b0bccef6e0e065020819da82127b2785d6a27f8c3516b36532`。
- 可执行模板 SHA256：`d8b863a32cd49ab26a7255ac7e3442160df1382bd0dd0ab0247474ab3ee1e4ae`。
- 源码身份、精确白名单和逐成员校验：同目录 `source-identity-check.json`、`files.txt`、
  `payload-manifest.json`、`archive-verification.json`。
- 最终测试证据：同目录 `check-002.log`。初次 34 项通过记录 `check-001.log` 保留。

准备时曾遇到“只读目录无法为新文件创建 mountpoint”的容器启动失败，尚未运行任何测试；
改用容器内存中的临时源码组合后通过，没有更改来源权限或 Docker 配置。
首次打包因跨 shell 变量传递失真失败，未生成归档；随后用明确相对路径完成上述包。

## 下一次 SSH 的具体动作

先只读核对交接指定实例、现有远端基础 workspace 与原始 CUDA 环境，并补取昨日最新
shutdown-request；不为该记录单独开机。确认基础 identity 后使用新 delta 创建新 workspace，
核验 93 项输入、上游代码和 20 个 checkpoint 文件，再进入原生 normalization/CPU 检查。
精确实例匹配沿用既有 Chrome 授权流程；不碰其他实例、充值或释放。

接收器与看门狗先于 worker 就绪；十条件合计 1,800 forwards、0 optimizer，工作最多
1,800 秒、关机截止 2,100 秒。原生及恢复对照任一失败立即停止，不改容差或重跑旧身份。
完成后回传全部结果、逐文件 SHA 校验、发送匹配收据、保护关机并核对平台状态。

下一科学读数必须来自这次原生 CUDA 干预。当前唯一根因仍未确定，M2 未完成，
Gate 3/4 无新测量，GPU 结果仍为 `not measured`。
