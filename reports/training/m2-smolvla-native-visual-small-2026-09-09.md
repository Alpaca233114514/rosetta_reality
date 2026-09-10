# SmolVLA native visual small-sample validation — 2026-09-09

**Outcome:** the eight trained images pass the visual-use check, while all four noise conditions fail on the five held-out scenes: mismatched images produce lower error. This is a negative generalization result. The visual problem is not considered repaired.

## Scope and external basis

The user authorized at most 30 minutes of server-only small-sample work. The [official SmolVLA guide](https://huggingface.co/docs/lerobot/smolvla) uses a 20,000-step, batch-64 fine-tuning example and recommends tuning training duration and starting with small batches. The [paper](https://arxiv.org/html/2506.01844v1) supplies the native frozen-VLM/action-expert recipe. This bounded check tests visual learning capacity with that native objective; it does not establish that additional full-dataset training will solve the task.

The latest historical 632-step, batch-32 run exposed roughly one 20,000-row train-set pass. Here, eight registered train frame-zero observations are repeated 128 times per anchor by the existing trainer. State, instruction and per-condition noise are identical across images. Five separate validation scenes receive no gradients. No paired auxiliary loss, state dropout, VLM unfreeze or new dependency was introduced.

## Measurement

Metrics use native ten-step denoising with zero noise and three fixed Gaussian seeds. A cyclic image mismatch preserves all nonvisual conditioning. The image-invariant comparator is the per-group mean target, an oracle lower bound for a prediction that cannot distinguish these same-state examples. Mixed-unit aggregate MAE remains only the historical registered proxy; joint and gripper errors are listed separately below.

| Split | Correct chunk MSE | Wrong-image chunk MSE | Image-invariant lower bound | Positive swap gaps | Centered cosine | Joint first-action MAE (rad) |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.026303 | 0.375370 | 0.192625 | 4/4 | 0.9522 | 0.022282 |
| validation | 0.217349 | 0.184712 | 0.121431 | 0/4 | -0.1277 | 0.021832 |

Registered trained-sample acceptance: **True**. Registered validation acceptance: **False**. Independent process reload: **exact prediction parity**.

## Engineering corrections and failure preservation

- Preflight now honors parsed batch size and checks explicit train-split episodes before creating the dataset.
- The v2 logger previously read unresolved parent YAML and raised `KeyError: tracking` before model loading. It now uses the resolved context and plan. Secondary Trackio cleanup errors no longer replace the original training exception.
- Cross-episode fixed samples use the existing sampler and native training loop. Their explicit identities are limited to bounded visual smoke plans; formal and resume usage fail closed.
- The first checkpoint checker rejected native BF16-to-FP32 upcasts. The recorded correction permits only exact-value-preserving conversion; all 345 frozen VLM tensors remain numerically identical. The original failure and checker are retained.
- Historical workspaces, plans, failed logs, recovery checkpoints and deployed artifacts remain intact. Nothing was committed, pushed, downloaded, or transferred as a model/data artifact.

Compute stages completed within 1197 seconds of the shared 1,800-second deadline. Total optimizer updates: 258 (two-step smoke plus 256-step pilot). CUDA allocation was capped at 12 GiB; measured resources and all result hashes are in the companion JSON.

## Interpretation and remaining work

This experiment covers only frame zero. A positive result on the eight training images is bounded visual learning evidence. Held-out frame-zero alignment is reported separately. Full-trajectory generalization and Gate 3/4 task success for this pilot are **not measured**; the historical M2 status is unchanged. No further training is authorized by this report. Shutdown has not been executed because the user explicitly conditioned it on repairing the visual problem.

Evidence: `runs/visual-native-small-001/`. Candidate plan: `configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml`. Recovery checkpoint remains under the registered durable checkpoint root and the plan run name.
