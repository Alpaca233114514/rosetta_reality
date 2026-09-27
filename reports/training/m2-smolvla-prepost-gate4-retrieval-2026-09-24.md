# Pre/post Gate 4 evidence retrieval and Basin history check

The finished AutoDL clone job `prepost-gate4-20260924-001` was read after a
user-approved, short no-GPU restart. The native `worker-exited.json` reports
`error: null`, zero optimizer steps and 826.342 seconds elapsed:

| Arm | Gate 3 | Gate 4 |
| --- | --- | --- |
| Pinned base weights under saved ALOHA interface | failed | not measured (Gate 3 prerequisite) |
| Canonical step-5000 weights | passed | failed |

These are native worker claims, not an independently recalculated Gate readout.
The base arm has no Gate 4 episode report; no success-rate comparison between
base and trained is established by this run.
The base Gate 3 report marks `joint_limits_respected` and
`maximum_unexpected_collisions` false. The trained Gate 4 report claims
success rate 0/5, 19 per-step joint-limit violations and 49 per-step
unexpected-contact counts; its failed criteria are those two safety limits
and `minimum_task_success_rate`. Counts are accumulated simulation steps,
not distinct incidents.

The complete small native evidence tree, excluding only linked
`pretrained_model/` files, was streamed read-only from the clone. Remote tar
SHA256 and independently computed local tar SHA256 both equal
`5a2886ab3815680a0482bfe7ade90400fb2d0db98b509693c5703c3aa01bacef`.
The archive is `.cache/prepost-gate-received-20260924-001.tar`; 44 regular
files were extracted under
`runs/prepost-gate-received-20260924-001/prepost-gate4-20260924-001/`.
The original remote files and the immutable local archive remain intact.

The user then narrowed the request to obtaining an existing Basin history.
A bounded read-only search under the clone's durable storage and common local
runtime paths found no `history/` directory or history `run.json`. The clone
contains the previously uploaded Basin **tool code**, not a Basin history
for this new run. This task's runner writes native Gate JSON and local Trackio
data; it did not invoke Basin import or a TorchLens capture. The latest
existing local Basin history, `rosetta-canonical-20260922-004/history`, has
seven units for the historical canonical training, 2500/5000 checkpoints and
Gate 004/005. It predates this base/trained comparison and cannot be relabeled
as its history.

The clone's no-GPU mode listed a rate of ¥0.10/hour with ¥0.01 minimum.
After the read-only check the platform shutdown was requested, and a fresh
instance-list reload showed the clone as `已关机`. No instance release was
requested. The source job's earlier `shutdown-request.json` is retained but
was not used as proof of the platform's final status.
