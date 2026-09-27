# Basin history deployed for the pre/post Gate comparison

The create-only deployment `prepost-basin-history-20260924-001` completed on
the 44126 clone in no-GPU mode. The uploaded converter SHA256 was
`5aa83717a42eb11f9f0c54eae8962f765619dfcea9ca30f1532651c7cabe32b9`.
It checked the four fixed native source-file hashes, worker outcomes, each
Gate report's plan/artifact binding, trained Gate 4's Gate 3 reference, and
the prior uploaded Basin code inventory before importing.

| Basin record | Status | Events |
| --- | --- | ---: |
| `prepost-gate4-20260924-001-base-gate3` | failed | 1 |
| `prepost-gate4-20260924-001-base-gate4-not-measured` | not_measured | 0 |
| `prepost-gate4-20260924-001-trained-gate3` | passed | 1 |
| `prepost-gate4-20260924-001-trained-gate4` | failed | 5 |

Basin verified all four histories. The model-callable surface exposed eight
read-only tools and no import tools. A Gate 3 comparison query found one
aligned summary event; it compares reported metrics, not identical physical
states. Remote stdio MCP initialize, tools/list and `basin_history` query
passed. The remote launcher is
`runs/prepost-basin-history-20260924-001/basin-mcp.sh`; it starts on demand
and has no persistent service or HTTP listener.

The complete remote history, launcher, converter and receipts were retrieved
as `.cache/prepost-basin-received-20260924-001.tar`. Remote and local tar
SHA256 both equal
`af5972a76afe3cdffb201ecfbcbe06de8b9e2ce7b64dc598c281ea3cfbdeb18c`.
The extracted copy is under `runs/prepost-basin-received-20260924-001/`.
Local PowerShell independently rehashed all 12 history files against their
four Basin manifests with zero mismatches. Local read-only Basin CLI listed
the same four records. The local stdio launcher
`scripts/mcp_prepost_basin_history.sh` completed initialize, tools/list and
history query after the server was off: protocol `2025-11-25`, eight tools,
all read-only, and four history results. It is available to a model host as
an on-demand command; this run did not edit user-level Codex MCP settings
or claim hot-loading in the current chat.

The source reports are preserved inside derived native envelopes with their
exact source SHA256 values. This is a historical report import, not a new
TorchLens capture or an independent certification of Gate acceptance. Base
Gate 4 has no rollout because base Gate 3 failed; it cannot be scored or
compared to trained Gate 4 success. The remote instance was shut down after
retrieval; a fresh platform page confirmed `已关机`. No instance release,
training, model inference, or hidden-test access occurred in this deployment.
