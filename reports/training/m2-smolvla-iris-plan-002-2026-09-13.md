# Iris 002: fresh run after metric-name repair

The user explicitly authorized repairing the smoke failure, opening the registered
AutoDL worker using Chrome, starting a new furnace and assigning Luna to monitor.
This is a fresh run, not a resume or retry inside the immutable failed 001 job.

- Run: `iris-k-scene-20260913-002`; names were searched before registration.
- Repair commit: `c308b8d`. Only public metric names changed from
  `image_key_scene_*` to `image_k_scene_*`; the public sanitizer was not weakened.
- Hypothesis and single axis: image K scene variance, lambda 0 versus 0.01.
- Keep the baseline recipe, model/data/processor revisions, Action Contract,
  train40 frame-0 split, batch 4, seed 20260809, AdamW and full-run 16-step warmup /
  1280-step decay unchanged. Use separate fresh two-step smokes and fresh 1280-step
  runs. Upstream scales smoke-only warmup/decay to 0/2; smoke state is never reused.
- Calibrate on fresh pinned base and training scenes, seal new calibration and
  all four plans. No old weights or optimizer state may initialize this run.
- Maximum 2564 optimizer updates; fixed 1280-step endpoint; preserve recovery
  checkpoints at 320/640/960/1280. No checkpoint search, hidden access or auto retry.
- Work/shutdown deadlines remain 3600/4200 seconds. Initial storage >=23 GiB,
  main admission >=17 GiB and actual smoke timings must fit. CUDA allocated <=8 GiB,
  reserved <=10 GiB, process tree RSS <=8 GiB. No budget expansion.
- Follow the exact prerequisite, export/reload, negative-result preservation,
  checksum/receipt and guarded shutdown chain documented in
  `m2-smolvla-iris-furnace-local-2026-09-13.md`. Source owners remain the same.
- Keep every 001 file and its failure report unchanged. Do not release the instance.

New template: `iris-furnace-template-002-2026-09-13.json`, SHA256
`35516b4302714971db57ef50da3dc14b53ee48bf40240f517ac6db2633beb1b2`.
The protocol, launcher, receiver and local preparation output all use 002.
Local fixed offline Docker checks passed: 118 tests, Ruff, 15 formatted files,
and individual shell syntax checks. The real metric-output sanitizer regression
is included. This registration is not a CUDA or scientific success claim.

Luna monitors the exact newly staged workspace. Once training is stable, sample
minimal status every five minutes using a separate control-shell `sleep 300`.
Notify the parent on failure, missing lifecycle protection, completion/package,
or required intervention. The parent handles verified result recovery and confirms
platform shutdown. Record and commit each completed module; no push or merge.

CUDA outcomes for 002 are pending. Main training, Gate 3/4 and M2 are not complete.
