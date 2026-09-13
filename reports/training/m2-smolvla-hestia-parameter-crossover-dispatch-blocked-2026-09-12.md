# Hestia 参数互逆实验：上传被自动审核阻止，未启动推理

用户提供当前 SSH 后，只读核验已通过：原 RTX 4090 D，显存 1/24564 MiB，
无计算进程，已有 Python 3.12.3，磁盘可用 6,363,480,064 bytes。
本地模板的 54 个身份再次匹配。

自动审核拒绝了原 staging helper 的整仓上传，理由是 SSH/实验授权没有明确
覆盖全部源码外传。随后完成了一个真正缩小的替代方案：只上传下列 14 个
本轮必要文件，其他源码从同一远端已使用的基线 workspace 内部复制到
全新目录；旧目录只读，增量解包禁止覆盖。

- `scripts/` 下六个 parameter-crossover 模块：参数互换、采集、守护、stream、
  verify 和 receive。
- 三个对应合成/门禁/传输测试文件。
- 本轮 plan MD 与 template JSON。
- 本地逐参数审计摘要 `result.json`。
- 先前从同一远端取回的 640、1280 两份 `arrays.npz`；仅含 normalized/
  standard predictions、targets、noise，不含原始图像或权重。

可审查的完整逐文件路径、大小和 SHA 在
`runs/hestia-parameter-crossover-dispatch-20260912-001/delta-inventory.json`。
最终白名单包是同目录 `delta.tar`，14 个普通文件，2,447,360 bytes，SHA：
`fd34c59724249d2d9d8ec21dd31fcec6ad4f5b8c9986f880d90d0a7a354c878d`。
没有连接凭证、私钥、整仓归档、原始图像或模型权重。

该缩小方案也被自动审核拒绝：仍要求用户明确授权这些具体文件向其提供的
SSH 目的地传输。因此两个传输命令都没有执行，未创建新实验 workspace，
未上传实验文件、加载模型或执行 forward/optimizer。没有绕过审核。

本地增加了 staging helper 的显式 `--verified-base-delta` 入口，以及
`scripts/stage_autodl_delta_from_wsl.sh`。它核对上传包 SHA/精确文件清单，
验证已有基线归档身份，在同一远端新建组合身份目录后仅添加白名单文件。
目前只有本地打包与语法证据，实际传输尚未执行。科学采集、参数替换、
数据身份和期限没有改变；原准备报告保留为历史状态。

为结束空闲计费窗口，按用户此前授权调用了远端既有、SHA 已核验的受保护
关机 helper；它仍检查 Trash、GPU 与其他任务。调用后 SSH 被远端关闭。
平台电源和计费状态未独立验证，实例未释放。没有给新的上传方案附加或
默认取得授权。

后续需要：用户明确允许将上述 14 文件包上传到其指定 SSH 主机，并在
带卡环境可用时继续。四组实验实际结果仍为 not measured，根因仍未确定，
M2/Gate 状态不变。
