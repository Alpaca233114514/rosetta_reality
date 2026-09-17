# Canonical post-training attempt 003

Attempt `canonical-fullframes-posttrain-20260914-002` passed 37 CPU checks but
failed the real CUDA doctor before any model load or Gate. The supervisor changed
`ROSETTA_RUN_ROOT` to an uncreated results directory and retained the old
`TRACKIO_DIR`. The doctor correctly rejected these inconsistent runtime roots.
This is an execution defect, not a measured model/Gate result.

The primary agent adds `prepare_output_roots`, creating the isolated results and
Trackio directories and setting both environment variables together with the
artifact root. Two regressions call the real doctor root validator and reproduce
both old failure modes. The doctor itself and its storage requirements are unchanged.

Fresh attempt `canonical-fullframes-posttrain-20260915-003` preserves the source
step-5000 endpoint, complete pretrained file seal, offline cohorts, Gate 3/4
protocol, 3600/4200-second budgets and independent shutdown protection. It uses
a new plan, template, workspace and output directory; attempt 002 is not retried
or overwritten. Luna may execute only after the primary agent verifies this revision.
