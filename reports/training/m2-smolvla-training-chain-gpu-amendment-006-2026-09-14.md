# GPU audit 005: native temporal recipe and affected metrics

The user requested rerunning affected measurements and auditing how code
actually executes and recognizes inputs, beyond inspecting data alone.
Run 004 passed no-optimizer parity, then its new ingress assertion failed
before optimizer: actual state per sample was [1,14], while the independent
audit query had loaded [14]. All 20 failure evidence members were recovered
and SHA-verified; completed optimizer steps remain zero.

Run 005 uses the pinned native resolve_delta_timestamps(policy.config, metadata)
for state, cameras and actions. It does not squeeze dimensions or change the
model's input. The independent raw comparison and real native trainer now use
the same declared temporal recipe. Observed model tensor shapes and runtime
observation/action delta indices are retained in ingress evidence.

Use a new source identity and remeasure affected parity, real two-step optimizer
smoke, saved-checkpoint full-array reload and fixed Iris control Gates. Original
thresholds, noise, sample schedule, 2700-second deadline and memory limits apply.
No formal furnace, model selection or hidden-test cohort is added.
