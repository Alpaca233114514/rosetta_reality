# Targeted dataset staging — 2026-09-27

Status: complete; payload independently verified and platform shutdown confirmed.

The user authorized direct downloads inside the existing no-GPU AutoDL
container, into the data disk, with discretion over selection under its 85 GiB
capacity. No dataset payload passes through the desktop. Existing experiment
outputs and dataset receipts are preserved. No training is authorized here.

Measured initial disk capacity: 91,268,055,040 bytes total,
13,151,170,560 bytes used, and 78,116,884,480 bytes available.
The sealed new selection is 66,158,775,113 bytes, with a 10,000,000,000-byte
reserve and approximately 11.96 GB expected remaining free space.

Durable staging namespace: `datasets_external/targeted-20260927-001` within
the existing remote Rosetta data-disk root. Previous verified sources remain
under `datasets_external/targeted-20260925-001`.

| Dataset | Revision or source identity | Selection | New bytes |
| --- | --- | --- | ---: |
| ALOHA insertion human | `cc571a3c661df81b566dbfde3d5c1e85fcdf7884` | Complete 11-file current-format snapshot | 91,348,537 |
| MimicGen | `33016f8a62c02334f929f2913af8fdd2a8a129e1` | square D0/D1/D2 plus README and attributes | 4,851,403,523 |
| Sirius RLDS | GCS per-object generations sealed in `plan.json` | Complete 67-object source listing | 7,031,694,115 |
| REBOOT USB-A recovery | `cfa3498a3982eb24554e88e77140247871eee3eb` | Complete 82-file snapshot | 37,283,634,402 |
| REBOOT RJ45 recovery | `f6acd5b69394ffd4d2d8bd30079306c9e2061bbe` | 24 data shards, all video shards and original metadata | 16,900,694,536 |

RJ45 data-file indices:
`0,3,5,8,10,13,15,18,21,23,26,28,31,33,36,38,41,44,46,49,51,54,56,59`.
Source episode metadata confirms 24 selected episodes and 21,559 frames,
out of 60 episodes; every selected episode's video reference is in the plan.
This is a partial dataset snapshot. Raw row counts and video decoding have not
yet been independently checked.
Original global metadata and statistics remain unchanged as provenance;
training must use an explicit local subset view and training-only statistics.
Remaining RJ45 data shards, RACER and AgiBot are deferred for capacity.

Previously staged ALOHA scripted (`8ab660912970111cbb26738b11458e6fc4a4aed1`,
70,994,713 bytes) and Can Paired raw/low-dim
(`74fa018461f479cd9fd15b924a16103012096203`, 100,568,305 bytes) are retained.
An older ALOHA human cache at the same revision also exists; the small current
snapshot is independently staged without altering that historical cache.

Execution uses eight remote HTTP streams, each reading at most 8 MiB
into memory at a time. Partial files are resumed, range responses are validated, and final
files require source LFS SHA-256, Git blob SHA-1, or GCS MD5 as applicable.
Each completed selection is independently reread after transfer. An interrupted
initial serial execution is retained in `download.log`; the intermediate
four-thread execution uses `download-parallel.log`. The active eight-thread
range-request execution uses `download-parallel8.log` and its own execution
receipt. The final active execution uses `download-stream.log`, preserving
all earlier records. Continuous per-file responses reduce repeated HTTP
overhead while bounded reads reduce memory. Console writes are synchronized
in the active worker to preserve JSON-line boundaries across threads.
Before increasing concurrency, the measured container limit was 2 GiB,
anonymous memory was 592,523,264 bytes, and OOM/OOM-kill counters were zero.
The large remainder of charged memory was reclaimable file cache; host-wide
memory totals were not used as the container budget.
The first streaming checkpoint had 27,121,703,659 payload bytes present,
no failed files, 461,099,008 anonymous-memory bytes, and zero OOM/OOM-kill
events. This is progress evidence, not final acceptance.

Evidence within the staging namespace: `plan.json`,
`parallel-execution.json`, `parallel8-execution.json`, `stream-execution.json`,
`download.log`, `download-parallel.log`, `download-parallel8.log`, `download-stream.log`,
`existing-verification.jsonl`, per-selection `rosetta_download_manifest.json`,
`rj45-subset-metadata-audit.json`,
and, on completion, `independent-verification-stream.json`,
`result-stream.json`, `exit-code-stream.txt`.

The previous scripted/Can Paired batch passed fresh source-manifest and
destination SHA checks: 12 files, 171,563,018 bytes. The RJ45 metadata audit
uses the existing remote runtime environment, without installing dependencies.
The initial metadata attempt encountered a local script named `inspect.py`
shadowing the standard library; that script was preserved by renaming it to
`source-inventory.py`, and the metadata audit then passed. Download execution
was unaffected. Transient SSH closures also occurred; the tmux worker continued.

## Final file acceptance

At `2026-09-27T03:41:21.273226+00:00`, the final worker had exited with code 0.
All 212 selected new files, 66,158,775,113 bytes, passed source-bound checks
and a separate whole-file reread. There were zero failed files and no remaining
partials. Together with the freshly verified old 12-file batch, the accepted
selection covers 224 files / 66,330,338,131 bytes. No dataset payload was
retrieved through the desktop.

Final disk observation: 91,268,055,040 bytes total; 79,312,007,168 bytes used;
11,956,047,872 bytes available. Container memory limit was 2 GiB, anonymous
memory 321,306,624 bytes, file cache 1,608,904,704 bytes, and OOM/OOM-kill
counters were zero. The worker ran in the registered platform container with
no nested Docker, model work or experiment task. No GPU doctor ran in no-card
mode; this is not next-stage GPU readiness evidence.

Only 14 metadata receipt files / 404,710 bytes were retrieved into
`reports/training/targeted-dataset-staging-20260927-001/`. Their available
source seals plus nine local worker-code files passed 22 SHA-256 comparisons;
the provenance index itself matched the separately observed remote SHA.
Code/log entries not included in this metadata-only copy remain on the remote
disk with their hashes in that index.

| Receipt | SHA-256 |
| --- | --- |
| `plan.json` | `7ca03b13c4f5647a6c411ec8975f746c920569b540be12e789ae34b4cb53bac6` |
| `independent-verification-stream.json` | `4eed7b1b31c86f768987e36a6005218cb315b8a38264b93fe1bf2ba4189ee003` |
| `result-stream.json` | `61942c315a7588af07cd36923786e50adb376ed85b6e7ab65f19ab6edef6c9d7` |
| `provenance.sha256` | `8f4fefad6f3955decba6c725a0122c2350676c697dc9e655371c7bb65ab3da2d` |

Raw row counts, video decoding, recovery/success labels and robot-contract
compatibility have not been established by this download acceptance.
The design and no-execution handoff are recorded separately in
`reports/training/three-experiment-design-2026-09-27.md` and
`reports/training/experiment-handoff-2026-09-27.md`.

## Platform closeout

After saving the metadata and handoff, the existing no-card instance was
shut down through the authenticated platform page. The shutdown dialog opened
at `2026-09-27T03:43:30Z`; confirmation had completed by `03:43:52Z`.
A fresh platform-page observation at `2026-09-27T03:44:55Z` independently
showed `已关机` for the same authorized instance. No release was requested.
The local metadata folder additionally contains
`platform-shutdown-verification.json`; a native screenshot is saved as the
local visualization artifact `autodl-dataset-stopped.png`. The earlier
`closeout-observation.json` correctly records the pre-shutdown phase rather
than asserting later platform state. No experiment task started.

Action contracts stay separate. REBOOT declares a bimanual WidowX follower,
14 named joint/carriage action dimensions, and 30 Hz. These are not the active
ALOHA 50 Hz contract even though both have 14 dimensions. Sirius, Can Paired
and MimicGen likewise require their own verified contracts. Source schemas do
not establish all units, control semantics or training compatibility.

Experiment 3 remains success-data versus recovery-data under matched counts,
frames and optimizer updates, with all other controls frozen. Downloading
these sources does not establish a compatible matched cohort, recovery labels,
a new experiment authorization, or improved Gate results. M2 is unchanged.
