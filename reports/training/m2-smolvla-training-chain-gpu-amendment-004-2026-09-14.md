# GPU audit 004: preserve native metadata boundary

Run 003 reached native update_policy but the new audit forward hook attempted
to read frame_index after the native processor had removed it. The hook raised
before loss/backward/optimizer; this is a diagnostic harness assumption, not
evidence that native training lost samples.

Run 004 observes actual episode/frame identities at the native cycle output,
without adding fields or replacing tensors. A pending identity list binds one
delivered batch to one policy forward; image, normalized state, full action
labels and padding remain independently checked at policy ingress. Existing
TrainingObservation still accounts for successful optimizer updates separately.
Preflight and smoke retain distinct names. All prior numerical, resource and
two-update bounds remain. This amendment does not claim GPU execution occurred.
