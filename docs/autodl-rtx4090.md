# AutoDL RTX 4090 worker

## Boundary

AutoDL is a temporary CUDA worker. Repository decisions and edits remain local;
the remote instance receives a versioned workspace plus immutable caches and
writes generated state to the AutoDL data disk.

The platform's container instance is already a Docker container. AutoDL states
that nested Docker is unavailable, so the instance itself is the formal Linux
container boundary. `scripts/run_autodl.sh` does not invoke Docker and records
`nested_docker_used=false` in its doctor evidence.

Relevant platform documentation:

- <https://www.autodl.com/docs/env/> — container and disk boundaries;
- <https://www.autodl.com/docs/ssh/> — SSH and key login;
- <https://www.autodl.com/docs/daemon/> — `screen`/`tmux` for long processes;
- <https://www.autodl.com/docs/cuda/> — image CUDA versus driver capability;
- <https://www.autodl.com/docs/huggingface/> — place `HF_HOME` on the data disk;
- <https://www.autodl.com/docs/save_money/> — no-card mode and shutdown behavior.

The local SSD has no redundancy guarantee, and an instance continuously powered
off for 15 days can be released with its data. Checkpoints and accepted exports
must therefore be copied back before relying on the instance as durable storage.

## Select the instance image

Rent one RTX 4090 instance and select an AutoDL PyTorch image that already has a
working CUDA build of PyTorch. Do not install CUDA, cuDNN, drivers or a different
PyTorch wheel manually. `bootstrap_autodl.sh` refuses a CPU-only image and pins
the preinstalled `torch` and `torchvision` versions while installing the
revision-pinned LeRobot code.

The profile is `configs/runtime/autodl_rtx4090.yaml`. It requires:

- exactly one CUDA device whose name matches RTX 4090;
- at least 23 GiB visible device memory;
- BF16 support through the selected PyTorch image;
- LeRobot `0.6.2` from revision
  `c903b114a90e703b3f7d0c46cb38727c328c55ff`;
- Trackio `0.28.0`;
- data, models, checkpoints, artifacts, metrics and compiler caches under the
  AutoDL data disk.

## One-time local setup

Add the instance SSH command shown by AutoDL to the WSL SSH config under an
alias such as `furnace`. Configure the public key in the AutoDL control panel;
never copy a private key or password into the repository.

From WSL Bash at the repository root, create a new versioned remote workspace:

```bash
./scripts/stage_autodl_from_wsl.sh furnace
```

The command transfers tracked plus untracked non-ignored source files. It never
deletes or updates an existing remote release. The printed workspace path is the
only workspace that should be used for that run.

Large ignored caches are separate and require an explicit transfer. Preserve
their revision-scoped directory names. The source roots below are examples of
the already-approved local caches; inspect them before running the commands:

```bash
rsync -a --partial --info=progress2 \
  data/lerobot_m2/lerobot--aloha_sim_insertion_human/ \
  furnace:/root/autodl-tmp/rosetta/data/lerobot--aloha_sim_insertion_human/

rsync -a --partial --info=progress2 \
  models/lerobot--smolvla_base/ \
  furnace:/root/autodl-tmp/rosetta/models/lerobot--smolvla_base/

rsync -a --partial --info=progress2 \
  models/hf_home/ \
  furnace:/root/autodl-tmp/rosetta/models/hf_home/

rsync -a --partial --info=progress2 \
  runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/normalization/ \
  furnace:/root/autodl-tmp/rosetta/runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/normalization/

rsync -a --partial --info=progress2 \
  runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/action-space-8a56bf3b42087938.json \
  furnace:/root/autodl-tmp/rosetta/runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/

rsync -a --partial --info=progress2 \
  runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/dataset_views/train-only-3e3c6b9d347e5e71/ \
  furnace:/root/autodl-tmp/rosetta/runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/dataset_views/train-only-3e3c6b9d347e5e71/
```

These commands do not use `--delete`. They do not authorize downloading a
missing dataset or model on AutoDL. A missing or mismatched manifest is a stop.

## First boot on AutoDL

Enter the versioned workspace printed by the staging command, then run:

```bash
bash scripts/bootstrap_autodl.sh
source /root/autodl-tmp/rosetta/envs/smolvla-cuda-001/bin/activate
bash scripts/run_autodl.sh doctor
bash scripts/run_autodl.sh benchmark
```

`doctor` loads no dataset rows or model weights. It verifies the GPU, package
versions, workspace identity, immutable model/VLM/dataset manifests, data-disk
roots and offline flags, then writes create-only evidence beneath the durable
run root.

`benchmark` reruns the registered train/validation baselines with
`hidden_test_loaded=false`. It runs before any optimizer step. The sealed test
episodes remain untouched.

## CUDA preflight and smoke gate

The generic preflight wrapper accepts only an explicit command and first repeats
the doctor and pre-training benchmark:

```bash
bash scripts/run_autodl.sh preflight \
  python scripts/run_autodl_preflight.py \
  --config configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml \
  --run-name <new-create-only-run-name> \
  --normalization-report \
    /root/autodl-tmp/rosetta/runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/normalization/train-only-3e3c6b9d347e5e71.json \
  --action-space-report \
    /root/autodl-tmp/rosetta/runs/m2-smolvla450m-aloha-insertion-action-repair-bounded-gripper-003/diagnostics/action-space-8a56bf3b42087938.json
```

The dedicated launcher adds a create-only CUDA peak-memory supplement while
leaving the checksum-bound historical Faust runner unchanged. Do not start an
optimizer after this merely because the forward pass is finite.
The first CUDA optimizer activity must be a separately identified two-step smoke
with batch size 1 unless a new CUDA performance plan registers and justifies a
different value. It must prove finite loss/gradients, measured CUDA memory,
checkpoint creation, independent reload, processor identity, Trackio durability
and no hidden-test exposure.

`bash scripts/run_autodl.sh formal` intentionally fails. After live RTX 4090
doctor, benchmark, forward and two-step smoke evidence exists, preregister a new
CUDA plan with measured batch size, forward/backward latency, peak memory,
optimizer/scheduler contract, checkpoints, maximum wall time and stop
conditions. The completed XPU Faust run is baseline evidence, not a resumable
CUDA checkpoint and not permission to silently change effective batch size.

## Long process and shutdown

Run a future authorized smoke/formal command inside `tmux` and redirect its log
to the durable run root. A disconnected SSH session must not own the process
lifetime. After the initial health check, the monitoring agent must block itself
with exactly `sleep 300`; after each wake it may take one bounded status sample,
then must return to `sleep 300`. Short polling, continuous `tail -f` and busy
waiting are prohibited. The sleep belongs to the monitoring shell, never the
training process, optimizer or dataloader. Never place `/usr/bin/shutdown` after
an unvalidated command chain:
first persist the exit status and logs, validate the export/checkpoint, copy the
required artifact back, and then shut the instance down from the AutoDL control
plane.

Gate 3 and Gate 4 remain local unless a future plan explicitly moves them. The
RTX 4090 should not stay powered on for the fixed Gate 4 wait intervals.
