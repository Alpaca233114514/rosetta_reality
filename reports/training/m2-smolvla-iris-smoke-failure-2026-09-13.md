# Iris CUDA smoke failure and metric-name repair

Run `iris-k-scene-20260913-001` stopped at treatment smoke, before main training.
Executed optimizer updates: control 2, treatment 1, main 0. No retry was attempted.
The failure does not establish a scientific outcome or change M2/Gate status.

## Actual execution

The registered RTX 4090D worker identity was verified using both SSH hostname and
the live AutoDL console. A new content-addressed workspace
`20260913T133558Z-be8bec5404eb-c5d344db9332` was staged from `be8bec5` plus the
existing preserved working tree. Its archive SHA256 was
`c5d344db9332592debb17ee265c16861a39c4b18f3fce8670ce53f3af1f5833a`.
The original template SHA was
`307567ad0a942fc3bb137550eabdeb50ec896de63c75e7cbab9008b77bccfcc1`.

CPU tests, CUDA doctor, real-data test (1 passed), benchmark, batch 1/4 forwards,
fresh-base calibration and plan sealing passed. Calibration used 42 forwards,
zero optimizer updates and reproduced loss/RNG/155 gradient tensors with lambda
zero. Control smoke completed two steps and saved a recovery checkpoint. Two
independent saved-processor reloads matched seven arrays exactly (38,800 scalars).
The underlying scheduler scales the two-step smoke warmup/decay to 0/2, as logged
by upstream; smoke is not evidence of full 1280-step scheduler behavior.

Luna was assigned monitoring at the user's explicit request and detected the
treatment smoke failure. The first treatment update was finite: total loss 7.375,
flow loss 7.366, scene penalty 0.864, reported gradient norm 50.355 and memory
about 2.22 GB. These are health observations, not task success. The error was:

```text
ValueError: metrics contains a forbidden sensitive key segment.
```

`image_key_scene_loss` and eight `image_key_scene_layer_*` metrics contain the
segment `key`. The existing public Trackio payload validator correctly rejects
that segment. This newly introduced metric naming conflict, rather than a CUDA
numerical failure, prevented the second treatment update. The supervisor stopped
progression after 309.839 seconds and did not enter main admission or training.

## Evidence recovery and shutdown

The parent retrieved `runs/iris-recovered-20260913-001/verified/` and verified all
49 files, 647,939 payload bytes. Archive SHA256:
`d93be8837772e5e219c954c54f84fdf750a47e2d01b3c0e3767ea24baf5e095b`.
Manifest SHA256:
`49e25ae12191fc880cd042154d19d030152c43c0367c88b2cf79e001ab4a91be`.
The verified receipt was delivered to the existing supervisor. The live console
subsequently showed the exact registered instance **powered off**. It was not
released. The shutdown request itself was produced after the archive and remains
pending retrieval at a future authorized window; do not reopen solely for it.
The control smoke weights/recovery state remain remote, with their complete
inventory recorded locally; this small failure archive is not a weight backup.

## Local repair

Only emitted metrics were renamed to `image_k_scene_loss` and
`image_k_scene_layer_*`. Feature/config identities and the mathematical objective
are unchanged. The public payload validator was not modified. A regression test
feeds actual regularizer output through the original sanitizer and checks that
all nine metrics survive, while a sensitive `api_key` field still fails.

The fixed offline Docker check completed **118 passed**, Ruff passed, all 15 files
formatted. The repaired code has not been rerun on CUDA. The original template
and failed remote workspace remain immutable historical evidence; their source
seal intentionally no longer matches the repaired local source. This run's
no-retry contract remains enforced. Any continuation needs a fresh registered
run/template/workspace identity, fresh initialization and the same prerequisite
checks; do not reuse this failed run or its smoke optimizer state.

Main training, treatment reload, paired scientific acceptance and Gate 3/4 are
not measured. M2 remains incomplete.
