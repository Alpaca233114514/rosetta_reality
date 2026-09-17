# Canonical post-training repair 002

The primary agent personally repaired the reviewed Luna adapter. This changes
post-training execution only; it does not alter the completed step-5000 weights,
training source, old reports or Gate thresholds. Luna executes the sealed result.

## Changes

- Resolve the parent experiment and export the full action-space dataclass and
  actual Action Contract, including `adapt_to_pi_aloha`.
- Bind every pretrained file to the recovered checkpoint seal before loading.
  Use saved processors without a new statistics override. Export starts with
  `candidate-manifest.json` and reload false; two new independent collections
  and exact comparison to the recovered native endpoint admit `manifest.json`.
- Filter train episodes in the Arrow reader before materializing baseline rows.
  Actual offline samples must match episode and frame offsets. Prediction receives
  observations only. Gripper statistics select the two encoded dimensions;
  inference latency excludes the separate fixed-flow loss forward.
- Render the actual Gate YAML from registered inference, resource, Gate 3 and
  Gate 4 sections; reject protocol or prerequisite drift before the native engine.
  Full native-versus-adapter chunks and unchanged parameters are checked again.
- Use an isolated result root and suffix 471. A detached supervisor and independent
  watchdog enforce 3600 seconds work / 4200 seconds shutdown, RSS/CUDA/storage
  bounds, and preserve failure evidence. No optimizer, checkpoint search or retry.

## Verified

CPU QA on the registered original AutoDL container, no GPU/model loading:
`runs/canonical-furnace-preparation-20260914-001/posttrain-cpu-qa-002.log`.
Ruff passed. All 37 targeted tests passed in 18.57 seconds. Counterexamples cover
the complete action-space constructor, altered model/processor/file sets, Arrow
hidden-row filtering, actual protocol drift, gripper dimensionality, independent
reload evidence, data-root restoration and state dimensions.

The real 1,201,362,478-byte pretrained directory passed full SHA/file-set validation.
Original training implementation validation passed in the new workspace. Endpoint
metadata preparation passed using the real checkpoint and dataset metadata,
producing 20,384 bytes while keeping reload false and no final manifest. No
checkpoint was copied or deleted. Durable free space after QA: 2,257,813,504 bytes.

QA workspace: `20260914T140951Z-95cf9cf9483b-66b3dba13657`; composite SHA:
`66b3dba1365706c2cbd151973ddca0835f2041cf09b7f305586bbd83ee5ccf8d`.
Plan SHA checked in QA:
`e012afd526118e86c89080c0fe3856281ba4aa5ca8feff2aa4a6d26ea2887810`.
The earlier lint failure and first QA logs remain preserved.

## Pending and limits

CUDA artifact reload, 160 train / 20 development offline observations under six
noise conditions, real simulation adapter bridge, Gate 3 and Gate 4 remain pending.
Gate 4 uses seeds 1000..1004, 500 steps each and the unchanged 0.2 success threshold.
A prerequisite exception is not a measured Gate failure. Gate 3 failure prevents
Gate 4. A negative measured Gate 4 is retained without retry or threshold changes.
Thirty-seven CPU tests do not prove model quality, full training generalization,
deployment success or M2. No simulator recordings are registered in this budget.
