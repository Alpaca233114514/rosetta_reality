# Repository agent guidance

## Repository safety

- Never delete, clear, overwrite, or discard existing files unless the user explicitly approves the exact affected paths.
- Before any deletion, list every affected path, explain the reason and a reversible alternative, and wait for approval. Broad requests such as cleanup, refactoring, or repair are not deletion authorization.
- If an operation might lose data and that cannot be ruled out, treat it as deletion and stop.
- Prefer compatible, incremental edits when an existing file or structure overlaps a task.
- Treat `main` and `master` as permanently protected branches.
- Put integration changes on a feature branch and use the normal pull-request workflow.
- Never force-push, rewrite Git history, bypass branch protection, or weaken required checks.
- Never bypass AutoReview.
- Do not commit or push unless the user explicitly requests it.
- Never commit secrets, access tokens, credentials, private data, or model access keys.

## GPU and environment safety

Never install or upgrade system-level PyTorch, CUDA, ROCm, GPU drivers, or simulators automatically.
Before any real GPU training:

1. Run `python scripts/check_env.py`.
2. Run the unit tests.
3. Run the dummy forward pass.
4. Run a CPU or small-model smoke test.
5. Run a short GPU smoke test.
6. Only then launch a real experiment.

Do not download model weights or datasets unless the user explicitly requests it.

## Architecture

- Keep the vision-language backbone replaceable.
- Keep the dataset layer robot- and embodiment-agnostic.
- Do not leak Qwen-specific behavior into generic VLA policy components.
- Do not hardcode action dimensions, chunk sizes, robot degrees of freedom, devices, or dtypes.
- Avoid speculative abstractions without an immediate use.
- Prefer a small working baseline before advanced research features.
- Imports must not load models, access the network, or mutate the environment.
