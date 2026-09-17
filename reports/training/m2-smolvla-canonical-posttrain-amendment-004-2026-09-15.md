# Canonical post-training attempt 004

Attempt 003 passed doctor, preparation, two independent collections, exact reload,
offline evaluation and Gate 3. Gate 4 rejected the matching-report prerequisite
before any episode measurement. Only the code tree SHA differed; the model
manifest and simulation-plan SHA remained identical.

The caller placed the live launcher log at the non-Git workspace root. The existing
code identity includes root files. Appending the supervisor's Gate 3 exit message
therefore changed the code identity before Gate 4. An in-memory counterfactual
removing only the post-Gate-3 log suffix reproduced the exact original Gate 3 tree
SHA `283e0d0d8ca8e3bd59c73e9352ffde5f94ec78e60719e32221dd569ae80b090d`.
Evidence: `runs/canonical-furnace-preparation-20260914-001/gate-identity-causal-proof-003.json`.
No historical log or Gate report was edited.

The primary agent changed the launcher to redirect all stage output itself to a
create-only `runs/<attempt>-launcher.log`. It no longer relies on the caller's log
placement. Two regressions exercise the real non-Git workspace identity: root log
growth changes it, durable run log growth does not, and actual source modification
still changes it. The Gate engine and code identity implementation remain unchanged.

Fresh attempt `canonical-fullframes-posttrain-20260915-004` uses the same step-5000
weights, saved processors, offline cohorts, fixed five Gate 4 seeds and thresholds.
It reruns the complete bounded no-optimizer chain in a new workspace/output root.
Attempt 003's successful Gate 3 and pre-Gate-4 failure evidence remain retained;
003 has no measured Gate 4 success rate. Finish 003 retrieval and platform shutdown
before admitting 004 so that the previous watchdog cannot affect the new worker.
