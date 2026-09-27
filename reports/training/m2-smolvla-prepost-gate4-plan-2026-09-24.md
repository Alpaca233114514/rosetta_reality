# Pinned base versus canonical step-5000 Gate 4 comparison

Run identity: `prepost-gate4-20260924-001`. This is a new, zero-optimizer,
create-only simulated comparison. The two arms differ only in the complete
`model.safetensors` file: pinned `lerobot/smolvla_base` revision
`c83c3163b8ca9b7e67c509fffd9121e66cb96205` versus canonical step 5000.
Both use the saved canonical ALOHA policy configuration, tokenizer,
preprocessor, postprocessor, train-only normalization, Action Contract,
camera map and simulation engine. The base arm is **not** a recovered
historical step-zero model. Training effects include the frozen weights'
BF16-to-FP32 serialization change; this experiment measures endpoint policy
behavior under a common task interface, not an isolated optimizer mechanism.

Bound weights are SHA256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`
and `d4c0d87cd8e66c723b07ec84f7f5e47875268bfd4e2057ec5683fdf56adcabef`.
The saved canonical artifact must be verified in full on the worker before use.
The original reports and model files are never edited. New artifacts consist
of metadata plus links to the existing files; output lives only under the new
run identity.

The target is the user supplied AutoDL clone at SSH port 44126. It was
observed with RTX 4080 SUPER, 32 GiB VRAM, CUDA driver 595.71.05,
16 CPU quota and approximately 62 GiB cgroup memory. Its default Python
does not have LeRobot; the existing `smolvla-cuda-001` environment has
torch 2.8.0+cu128, LeRobot 0.6.2, gym-aloha 0.1.4, Trackio 0.28.0 and
MuJoCo 3.8.1. An old workspace's `/mnt/d` link is broken on this clone;
the verified canonical checkpoint is available under the durable
`checkpoints/` root. The old RTX 4090 profile is not used as proof of this
clone's GPU identity. A newly staged source workspace also inherits a broken
`runs` link to `/mnt/d`; this run writes under the clone's verified durable
`runs/` directory instead. The unused first source staging is retained.

Before Gates, each arm must load in two independent processes and yield
exactly identical complete zero-noise actions on the fixed Gate 3 initial
observation. This is a sampled reload check. The native Gate engine then
runs Gate 3 with seed 20260809, at most 20 steps, followed only if passed by
Gate 4 with seeds 1000–1004, 500 steps each, matching policy-noise seeds,
receding-horizon first action and unchanged 0.2 success threshold. Original
action limits, projection, collision policy and other safety criteria apply.
Trackio remains local; no hidden tests, model download, optimizer step,
training or physical robot operation is authorized.

One detached supervisor owns the whole two-arm sequence. Its independent
watchdog stops the child and supervisor at 5400 seconds and requests platform
shutdown by 5700 seconds. The 90-minute work limit allows roughly twice the
historical 18–19-minute one-arm post-training runtime plus clone variability,
reload and setup. The supervisor also stops on model/code identity drift,
another GPU user, reload mismatch, runtime error, >12 GiB process RSS or
<1 GiB durable free space. Gate 3 failure records Gate 4 as not measured.
Historical failures and any new incomplete outputs are preserved; there is no
automatic retry or checkpoint substitution. Results remain on durable storage
for later retrieval and verification, without live agent monitoring.
