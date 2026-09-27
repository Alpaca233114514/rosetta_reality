# 工具传输与 Google Drive 归档收尾

## TorchLens / Basin

已按仓库正式 `stage_autodl_from_wsl.sh --verified-source-delta` 入口上传
Basin 源码与已审核的 TorchLens/graphviz wheels，未传模型或实验数据。
远端新工作区相对 Rosetta durable root 为
`workspaces/20260923T110402Z-2687d7a884d4-8684002da746`。

- 工具包 SHA256：`5303f250a485e3ec82b2940329eb922653a65e76f3f753283188dcf07bb68fef`。
- 基础工作区标记：`6bb3da3027dfc66fe876551977cdbaedd2f217f877dc4386f7f4269db538c96e`。
- composite identity：`8684002da746383fe085a7ddfb48c15d8c73d1874296b935884d0e443d0bee0b`。
- Basin 位于新工作区 `tools/basin`；依赖以离线、`--no-deps --require-hashes`
  方式展开在独立 `runs/prepost-tool-overlay-20260923-002/overlay`，原训练环境未改。
- 远端验证所有 payload manifest 文件 SHA，通过 Basin backend 惰性导入检查，
  CLI 工具发现完成，metadata 确认 TorchLens 2.23.0、graphviz 0.21。
- 未构造模型、未执行 TorchLens 真实 trace、未写入已有 Basin history；
  安装/传输完成不代表真实 SmolVLA/CUDA 兼容性已经验收。

本地记录：`.cache/prepost-tool-stage-001/stage-002.log`、`install-002.log`、
`payload/tools/payload-manifest.json`。远端 `check.json` SHA 为
`46b0178f93864ed48cee8333d3d24e7623edb881eddc2c36fab4b226194a36e7`，
`basin-tools.json` SHA 为
`4eb810861354e28b2cfeb28f920b8b5b97af05a99d7ba2a8b6feb57ef23371d7`。
初次本地打包路径错误和首次跨系统 shell 变量引用失败均未覆盖历史数据；
前者 0 字节输出已移入本轮缓存，修正采用新归档及新隔离输出身份。

## 归档停止边界

按用户要求使用 Sol / medium 子代理，盘点系统盘/数据盘后先迁移再校验，
没有删除任何远端源文件。子代理发现符号链接/硬链接和大量唯一训练历史，
没有把看似重复的目录直接认定为可清理空间。

用户后来明确“数据不需要过本地”，因此立即停止本地中转方案。变更之前：

- 已将 12 个旧工作区 tar 复制到本地
  `runs/storage-drive-archive-20260923-001/workspace-archives/`，保留全部副本。
- 仅 1/12 个 tar 上传 Google Drive 并完成完整字节回读 SHA 校验。
  其余 11 个未上传，12 个远端源全部保留。
- 已验文件 `20260814T141210Z-5bd66d5e4bdc-af0e415d6479.tar`，
  2,949,120 字节；子代理核对源/本地/云端回读 SHA 均为
  `af0e415d64794b9b2ecd11537d7687150b9168040e7a4f74bdf2be5526a7279b`。
- 本轮 Drive 目录：
  https://drive.google.com/drive/folders/1yZbZdVtmdeE2U4MP2REFsp2jxq5psBG_
- 已验文件 ID：`1nmf_yWC0FVFStbzNm5kRB-hgacQubOgy`。

直传检查没有发现远端现成的 rclone/Drive 授权或 Drive 挂载。
用户先同意准备官方客户端并自行 OAuth，随后要求先测试连接，并明确
“如果连不上，就关掉子代理不进行上传”。远端连通性结果由子代理报告：

| 端点 | DNS 独立查询 | HTTPS 结果 |
| --- | --- | --- |
| Google Drive API (`www.googleapis.com`) | 返回 IP | curl 28，约 10.009 秒，HTTP 000；请求内部解析超时 |
| Google OAuth 授权 (`accounts.google.com`) | 返回 IP | curl 28，约 10.002 秒，HTTP 000；连接超时 |
| Google OAuth token (`oauth2.googleapis.com`) | 返回 IP | curl 28，约 10.001 秒，HTTP 000；连接超时 |

这证明当前测试路径没有建立 HTTPS/HTTP 通信；不是收到 401 的未授权状态，
也不是断言 Google 服务全球不可用。没有改代理、防火墙或复制凭证尝试绕过。
子代理已按用户指令结束，上传/安装/OAuth/重试全部停止。
网络检查要求前启动的 rclone 下载在约 544 KiB 时中止；未完成校验、未解压、
未运行，部分文件保留为失败证据。没有使用该二进制。

主模型侧仍是无卡只读前置核验完成；真实 forward/backward/Gate 未运行，
见 `m2-smolvla-prepost-remote-preflight-2026-09-23.md`。实例关机最终状态
记录在本报告的 JSON 收据中，不能仅凭关机请求或 SSH 断开推定。
