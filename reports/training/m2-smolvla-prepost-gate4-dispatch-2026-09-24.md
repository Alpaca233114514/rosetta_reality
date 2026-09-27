# Pre/post Gate 4 dispatch on the AutoDL clone

Task identity: `prepost-gate4-20260924-001`. The user authorized one-shot
deployment to SSH port 44126 and requested no continuous monitoring.

Source workspace: `workspaces/20260924T110713Z-2687d7a884d4-c3ee2980bb5b`.
Workspace archive SHA256:
`c3ee2980bb5b4252557782e4251b7aa19fb266c97f8e59299e9a790aa83b1c57`.
Runner SHA256:
`ce32b229f56200621e2b6e1d942f8b6388635dfb1dd0910d3086c8aff5784384`.
Launcher SHA256:
`ba2be2abba05ce7cad601a2f7e52b2c46a1cbec9c7dfffb91a8fbd646ffd479d`.
The first staging identity `20260924T110342Z-2687d7a884d4-2ea8c474d9c9`
was not launched; its inherited `runs` link points to a nonexistent mount
on this clone. It remains preserved. The launched job uses the real durable
`runs/prepost-gate4-20260924-001/` directory.

At the single launch confirmation, the create-only `registration.json`,
`watchdog.json` and `preflight.json` existed. The preflight status was passed.
Supervisor PID was 2124; independent watchdog PID was 2139. Both were alive;
the active child was the base arm's first independent reload probe.
Registration started at 2026-09-24 19:09:21 CST. The work deadline is
2026-09-24 20:39:21 CST; the protected shutdown deadline is
2026-09-24 20:44:21 CST. The watchdog targets only the registered child and
supervisor process identities before platform shutdown. The shutdown request
is not proof of platform billing state; confirmation requires a later platform
status check.

No new Gate result was available at dispatch. The job should leave
`runs/prepost-gate4-20260924-001/worker-exited.json`, per-arm Gate reports
under its `results/`, logs, and a shutdown request. Base Gate 4 is measured
only if base Gate 3 passes; the same applies to the trained arm. No optimizer
steps, new model download, retraining, hidden-test access or physical-robot
execution are in this task.
