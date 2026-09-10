# Local temporal visual-grounding diagnostic — 2026-09-09

Registration: the JSON companion freezes source checksums, two local deploy
manifest checksums, samples, interventions and resources before data/model
execution. The user authorized local exploration and repair, and selected
local execution when asked about the Docker blocker. This does not authorize
another furnace, SSH, downloads or a Gate protocol change.

The concrete measurement gap is that the recent vfunfreeze formal plan selects
checkpoints on only frame 0 of five validation episodes. Those states are
identical. That slice cannot determine visual grounding later in a trajectory.
Historical plans, checkpoints and reports remain unchanged; the September 6
visual-unfreeze weight-update proof and 0/5 Gate result remain evidence for that
candidate only. Neither result rules out every visual-learning hypothesis.

This diagnostic holds each destination's proprioception, instruction, action
target and diffusion noise fixed while swapping only the image with the
preceding episode at the same frame offset. The five train episodes are
49, 4, 23, 43 and 21; offsets are 0, 100, 250 and 450. Each pair uses zero noise
and Gaussian seeds 20260905, 20260906 and 20260907. Each artifact receives
160 forwards at batch 1, with a 20-minute inference budget, inside the pinned
local Linux/XPU Docker image, networking disabled and memory/swap capped at
6 GB. No optimizer or simulation is involved.

Labels are filtered by episode and offset at the Arrow scan boundary. A
separate `pytest -m data tests/test_smolvla_visual_grounding_data.py` checks
the same 20 decoded observations against those rows without an optimizer.
The existing generic data smoke is deliberately not used: it constructs an
optimizer and requests all configured episodes, including the sealed split.
Read-only cache inspection and checksum validation precede weight loading.

Reports retain sample/image identity, source checksums, camera masks and
SigLIP input ranges, predictions, correct/mismatched MAE, paired per-episode
gain, correlations, physical-unit groups and the state-persistence baseline.
Every offset/noise result is reported, including negative gains. Non-finite
inputs/actions, source drift, unavailable runtime, OOM and budget overrun stop
execution with the failed log preserved. No retries follow OOM.

Interpretation is restricted to these training slices. Mismatched images may
contradict the destination robot pose away from reset; a positive or negative
gain alone is not a task-success or representation-absence verdict. Mixed-unit
aggregate MAE remains a proxy. The vfunfreeze-003 weights are not present
locally, so these two local controls cannot retest the latest candidate.

The accompanying implementation repair validates the entire non-visual VLM
before changing `requires_grad`, rejects empty scope modules and shared
parameters before mutation, and validates every checkpoint wrapper before
installing any wrapper. The previous checker missed language-model namespaces,
token embeddings and the VLM output head, and its alias error left already
changed parameters trainable. These are guard failures, not proof that the
historical run trained the wrong scope; its independent tensor comparison is
retained. Future training must register the changed implementation hashes.

Initial verification before data execution: 63 focused tests passed inside the
registered image; Ruff check and format check passed. These tests do not
establish improved learned visual grounding.
