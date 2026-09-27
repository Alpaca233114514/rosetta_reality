# Basin deployment for the pre/post Gate comparison

Deploy identity: `prepost-basin-history-20260924-001`, create-only on the
AutoDL clone used for `prepost-gate4-20260924-001`. The goal is a queryable
Basin history and read-only stdio MCP entrypoint for the completed Gate
reports. No model, dataset, simulator, TorchLens, optimizer, or GPU runs.

The previously uploaded Basin source under
`workspaces/20260923T110402Z-2687d7a884d4-8684002da746/tools/basin`
is the tool implementation. Its code inventory and bytes must match the
uploaded payload manifest. The source evidence is the immutable native
Gate report tree under `runs/prepost-gate4-20260924-001/`. Exact source
report SHA256 values are fixed in
`scripts/deploy_prepost_basin_history.py` and checked before writing a new
history. The runner also binds each report to the corresponding Gate plan
and artifact manifest, verifies the trained Gate 4 reference to the trained
Gate 3 report, and rejects any base Gate 4 report because its Gate 3 failed.

The deployment creates four Basin native-envelope records: base Gate 3,
trained Gate 3, trained Gate 4 with five seed events, and an explicit
base Gate 4 `not_measured` record. Each record retains the full source
report, its SHA256 and an evidence pointer; the envelope is a **derived
historical import**, not a fresh model capture. Basin's history/verify,
Gate 3 comparison, and read-only MCP initialize/tools/list/history call
must pass. Its launcher provides eight read-only tools and exposes no
import/source alias to model calls. The remote MCP is stdio on demand;
it does not keep a network service running after instance shutdown.

Use the user-approved no-GPU mode (0.5 CPU, 2 GiB memory, no GPU), with a
10-minute task timeout and no automatic retry. Preserve failed partial
directories. Download the new history as a separate tar, compare remote
and local SHA256, then request shutdown and verify the platform is off.
The new history can answer which registered report fields differ; the
base Gate 4 has no rollout, so the history cannot establish whether the
untrained base would pass Gate 4 or whether training caused its failure.
