# 原 D train40 自含 native 视图：有界 pilot 实测

本报告仅验收原生数据入口的 CPU pilot，不表示三实验训练完成或 M2 成功。全40任务 `d-native-full40-20260927-007` 已单独登记，结果另行验收。

## 输入和隔离边界

标签来自已独立审计的 train-only 导出 `train-00.parquet`，SHA `c55ae3ec2c93541dfee7ac611e9c20a3577b666df1cd15501a6e0db4d40c6187`；包含原 D40×500 帧。source receipt SHA `97cce7b398ef79139109e5cac02cdfb2aeef8edf2c3360109c1c92fca93a34d5`，revision `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`。本轮 builder 未重新打开原含保护集的 mixed data Parquet。

原视频只有一个共享 AV1 文件 `videos/observation.images.top/chunk-000/file-000.mp4`，88,389,634 bytes，SHA `25c0aca82aa50c815d7e5b25b4eebd70f7e61ea9d563dff7c39bda139cb935a3`。episode49 的 PTS 区间为490–500秒。隔离 builder 的向后 keyframe seek 可能在内存瞬时解码其他 episode 的 GOP 预滚帧；不保存、分析或作为训练样本导出它们。不能宣称底层从未解码保护帧。仅 train 白名单的指定 PTS 帧写成无损 PNG；正常 native consumer 只读新自含产物，无原 mixed shards/shared video 依赖。

D-only normalization 原文件 SHA `263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988` 保留，后续自含产物逐字节复制该报告；不以 global stats 或新视图冒充旧 view manifest。

## 实测结果

| 新任务 | 实际阶段 | 帧数 | 时间 | 峰值 RSS | 结论 |
|---|---|---:|---:|---:|---|
| pilot003 | episode49 隔离 build | 500 | 9.3253 s | 217,739,264 B | 成功，输出26,432,067 B |
| read004 | 原 upstream action-column native read | 500 | 184.4775 s | 1,339,850,752 B | 全像素/状态/50动作 chunk 精确 |
| read005 | instance action tensor-cache native read | 500 | 10.6839 s | 911,200,256 B | 同输入全部精确，约17.27倍加速 |
| two-episode006 | 原IDs4/49 native read | 1,000 | 14.4174 s | 984,588,288 B | 两物理nativeIDs0/1分别映射原4/49，全部精确 |

所有实际 read proof 都独立从标签 ledger 的原数组重算 `min(frame+i,499)` 动作目标和 `min(50,500-frame)` 有效 mask，核验末49帧 padding。全部当前帧像素与隔离 builder 的 RGB byte SHA 精确一致，状态和50×14动作逐项精确相等。两episode读取同时覆盖 episode 边界；原 hf_dataset episode 列和返回 batch episode 都保留原ID，不添加新 batch key。meta 描述真实物理episode数，没有伪称仍为50。

优化只替换具体 native reader 实例内的 action 列重复转换，cache 只来自 SHA-bound 自含 train Parquet 的 index/action。原 native reader 仍计算 chunk 查询绝对索引/padding、读取当前图像/状态/task；没有全局 upstream patch、模型 forward、optimizer 更新或 GPU 计算。

## 库存和准入

原D保持20,000个 padding-aware 输入帧：951,000有效目标 slots、49,000 padding slots；另披露18,040个 full50 输入，902,000全有效目标 slots。两套库存均保留，没有为跑通训练静默删除尾帧或改 loss。

`load_native_training_view` 与实际 `install_native_D_factory` 使用同一 admission predicate：真实full40/20,000帧、原ID全集、全部像素和state/chunk精确证明、原D-only normalization SHA；factory 额外绑定证明文件 SHA。pilot 的单/双episode视图不会进入正式训练。7项实际 CPU stdlib 测试覆盖这两入口门禁及 exact restorer。

全40预估约766.6MB、build约6.2分钟；native read 的线性保守估计约5分钟，属于预测，不是实测全40成功。注册007以2100秒总预算、build/read各1000秒、6GiB RSS、1GiB生成物、至少2GiB空盘、独立guard执行；科学plan SHA `46ba376e471583b5dde723f9e6caa6c0304c8be278e28126cd8c33324235ac0b`。

## 证据和失败保留

`pilot001` 因缺少 `/usr/bin/time` 在解码前退出127；`pilot002` 误传历史cache根缺原下载receipt，在视频解码前退出1；`pilot003` build成功，但初次read的6GiB虚拟地址限制误伤OpenBLAS。后续read004使用实际6GiB RSS守护及固定2线程，保留所有旧证据和视图，无重跑旧身份。

原native read004关键manifest/proof/resource/registration/exit五文件双端SHA一致；archive SHA `bea857d583d8dc2fcb37264a65fcd0d7c6ed0710c6abd4a07e3254ea8d8efe23`。其证明 SHA `0dd93134c6d6a4f076dda9ca77f2d5f8499721f2a631af8b4493de5958128cee`，绑定 view manifest SHA `b1d2639636a48c6d9de053a7176158b5309662ee25aa1cc6eda4052b06d1f9f6`。

cache005及two-episode006共25个small文件全部独立双端SHA校验通过；archive SHA `92d1df51ff55f11a40b8ea6da2342fae6ae47a45052e29ce80f312edb9bd36ed`。接收目录分别为 `.work/d-native-pilot-20260927-004-independent-received` 和 `.work/d-native-cache-20260927-006-independent-received`，没有取回原图像或大型Parquet。

guard worker/read groups 已证实退出。所有阶段均未请求关机、提交、推送或删除远端数据。