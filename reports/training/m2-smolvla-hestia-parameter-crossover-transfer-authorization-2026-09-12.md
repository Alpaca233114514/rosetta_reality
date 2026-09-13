# Hestia 参数互逆实验：具体上传已获授权，等待恢复 SSH

用户对上一个具体授权问题回复“允许”。该授权覆盖已列明的 14 个文件向
此前指定 SSH 目的地的传输：2,447,360 bytes，SHA
`fd34c59724249d2d9d8ec21dd31fcec6ad4f5b8c9986f880d90d0a7a354c878d`。
本地重新核验了包身份，没有变化。此范围内不再重复索取上传授权；执行时
仍通过正常权限申请和 Auto Review，不关闭或绕过审核。

随后的只读 SSH 探测被远端关闭，未上传、未创建新实验 workspace、未启动
模型。连接关闭不独立证明平台电源或计费状态。当前缺的是可用的原 4090 D
带卡 SSH，而不是上述具体文件的用户授权。

静态复核同时发现私有 `launch-delta.sh` 中保留了一个没有输入流的旧 tar
解包命令。已在新文件 `launch-delta-ready.sh` 删除该残留，并以新
`current-delta-ready.sh` 调用；原文件保留，未修改历史阻塞证据。Bash 语法
检查通过。该修正只涉及本机调度，不改变已经授权的 14 文件包或科学协议。

机器记录见 `runs/hestia-parameter-crossover-dispatch-20260912-001/explicit-upload-authorization.json`。
后续使用同目录 `current-delta-ready.sh` 继续。实际四组 CUDA 结果仍为
not measured，唯一根因尚未确定，M2/Gate 状态不变。
