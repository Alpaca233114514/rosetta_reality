# 原 D train40 自含 native 视图：全量实际验收

原 D 的40个训练 episode、20,000帧已完成隔离图像导出和真实 LeRobot CPU读取验收。正常训练入口不再需要原含保护集的 mixed Parquet 或共享视频。这是数据输入能力验收，不代表本轮三臂训练/闭环任务完成或 M2 成功。

## 固定身份与实际运行

- 科学计划：`configs/diagnostics/d_train_native_view_20260927_007.json`，SHA `46ba376e471583b5dde723f9e6caa6c0304c8be278e28126cd8c33324235ac0b`。
- run：`d-native-full40-20260927-007`；source workspace `20260927T134945Z-66a5bbcdbdcd-2d6110ac8761`，composite SHA `2d6110ac8761e45923737941227f190550a372f6451d58e2f5cab8a892af7c73`。
- 独立guard注册总2100秒，build/read各1000秒、RSS6GiB、生成物1GiB、最低空盘2GiB、CUDA设备不可见。build/read实际退出均0，worker/guard owned group均证实为空，未请求关机。
- build实测438.969秒、峰值RSS232,849,408bytes；native read实测193.554秒、峰值RSS1,867,190,272bytes。
- 自含payload共757,114,462bytes（manifest生成前计数，不含最终manifest/read proof）；包含40个PNG-inline train Parquet、纯train metadata、逐帧pixel/label身份、两套库存、原D-only normalization报告。没有模型加载、模型forward或optimizer更新。

## 输入与原生合同

revision固定 `cc571a3c661df81b566dbfde3d5c1e85fcdf7884`。state/action仅来自独立审计train-only导出SHA `c55ae3ec2c93541dfee7ac611e9c20a3577b666df1cd15501a6e0db4d40c6187`，未重读原mixed data shards。source receipt SHA `97cce7b398ef79139109e5cac02cdfb2aeef8edf2c3360109c1c92fca93a34d5`，isolated export manifest SHA `f94a629e0e3e9301d66cdef8a00aad1e88d8d61d8069bd505bd42d037ca878cc`。

source只有一个共享AV1视频；backward keyframe seek可能在隔离builder内瞬时解码保护episode的GOP预滚帧，不保存、不分析或作为训练样本导出。仅train白名单的准确PTS帧写无损PNG。不能宣称底层从未解码保护帧。正式consumer只读取已物理隔离的train产物。

metadata真实声明40 native episodes和20,000frames、50Hz、camera `observation.images.top`、image dtype及无video path。内部0..39按原episode排序映射到实际原40IDs；选择episodes、hf_dataset episode列、getitem batch身份全部桥接。原schema保留14维state/action和标准LeRobot keys，未增加破坏paired模板的batch provenance字段。

D-only normalization原SHA `263880ec3adfddb8517a50fa5483e7c8f32f0c208243c229cbc72f8e9cf8d988` 逐字节复制到新root `normalization.json`。meta.stats使用其effective state/action统计及固定ImageNet视觉常数，不重算跨集合统计，也不冒认旧normalization view manifest。

## 全量真实数值验收

真实reader逐帧读取全部20,000个训练帧：

- 每个原episode/frame身份一致，像素逐字节SHA与隔离导出的RGB一致。
- 每个14维state精确一致；所有50×14动作chunk与独立label ledger原数组的clamped索引精确一致。
- 每episode最后49帧padding数量和endpoint动作重复正确，chunk不跨episode边界。
- 40个完整原episode无保护集行，finite state/action通过。

native reader原本每frame重复转换HF action列。仅对具体dataset实例的查询使用来自该SHA-bound自含train Parquet的index/action tensor cache；原native helper仍计算绝对chunk查询索引/padding，原native图像/state/task读取保留，没有全局upstream patch。先前500帧unmodified native与cache均对同一原label数组逐项精确；性能由184.477秒降至10.684秒。原IDs4/49双episode1,000帧真实pilot也全部精确。详见 `reports/training/d-train-native-view-pilot-result-2026-09-27.md`，旧失败和旧proof均保留。

## 正式训练入口交付

独立模块 `src/rosetta_reality/diagnostics/d_train_native_view.py` 提供 `load_native_training_view` 和 `install_native_D_factory`，两者统一要求full40/20k实际读取证明、原ID全集、全部pixel/state/chunk精确及原normalization SHA。actual factory还绑定native proof SHA，恢复器要求wrapper ownership未变化。7项真实CPU stdlib tests通过。专用paired launcher由集成代理负责，未修改旧launcher/norm历史行为。

固定view manifest SHA `019e63cf967c144cc283b40c0cb99f5599c6b2581e5cd54df9f7f37aded656fc`；native proof SHA `39bdb7633bed288796af945050c2e10ff0aad8c6306d0d7866a076346180ec0b`。proof明确记录 `native_action_query=per_dataset_verified_train_only_tensor_cache`，status passed，original40、native40、read20,000。

## 独立回传及曝光库存

30个small文件逐一双端SHA一致；接收目录 `.work/d-native-full40-20260927-007-independent-received`，archive SHA `a4c137c7d0a1943d21d52a736cd96b67ac549467bd9ba7bdf7bd70f6591a90f6`。没有回传原图像、大Parquet或label值。8个已接收manifest成员在本地再次对manifest SHA核验；其余42个成员由真实远端reader入场逐文件重hash，未冒称本地已收到完整数据。

独立脚本 `scripts/verify_d_native_view_result.py` 对接收日志、metadata、inventory和pixel hash账进行实际独立重算，通过结果在 `reports/training/d-train-native-view-independent-verification-20260927-007.json`：

- 原padding-aware输入20,000帧：951,000有效target slots及49,000padding slots。
- full50输入18,040帧：902,000全有效target slots。
- 20,000个唯一pixel身份项、物理native与原IDs映射、40真实metadata、norm SHA、两阶段rc0及worker/guard退出全通过。

两套库存均保留，不静默丢弃尾帧、不修改native loss。正常正式训练是否选择原20k或全valid18,040，须遵循本轮已注册共同实验口径。全D数据准备已交付；三实验真实训练、checkpoint/resume/export/reload及闭环结果仍由各自阶段验证，不能以本报告推定完成。