"""Opt-in native DataLoader composition for the sealed paired-data experiment.

The original D metadata/statistics and native loss remain unchanged. Imports of
torch and numpy occur only inside the registered runtime installation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _verified_json(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Paired runtime JSON identity changed")
    return json.loads(path.read_text(encoding="utf-8"))


def scheduler_horizon_config(cfg,registered_updates):
    """Construct the native shared schedule without changing the loop stop."""
    import copy
    if registered_updates!=5000 or cfg.steps not in (2,5000):
        raise ValueError("Paired schedule horizon or actual stop differs")
    value=copy.copy(cfg)
    value.steps=registered_updates
    return value


class PairedMixtureFeature:
    name = "paired_mixture"

    def __init__(self, parameters):
        if set(parameters) != {"schedule", "schedule_sha256", "auxiliary_root", "auxiliary_inventory", "auxiliary_inventory_sha256", "camera_map"}:
            raise ValueError("Paired mixture requires complete schedule/data identity")
        self.parameters = dict(parameters)

    def install(self, context):
        import torch
        import numpy as np
        import lerobot.scripts.lerobot_train as module

        from rosetta_reality.vla.training.paired_mixture import digest

        root = context.plan_path.parent
        p = self.parameters
        def cuda_budget():
            if getattr(context,"device",None)!="cuda":return
            limits=context.plan.get("paired_cuda_budget")
            if not isinstance(limits,dict):raise ValueError("Paired CUDA budget must be preregistered")
            if (torch.cuda.max_memory_allocated()>limits["maximum_allocated_bytes"]
                    or torch.cuda.max_memory_reserved()>limits["maximum_reserved_bytes"]):
                raise MemoryError("Paired CUDA measurement exceeded its frozen allocation/reservation budget")
        def path(raw):
            value = Path(raw)
            if value.is_absolute() or ".." in value.parts:
                raise ValueError("Runtime declarations must be relative to plan directory")
            return root / value
        schedule = _verified_json(path(p["schedule"]), p["schedule_sha256"])
        entries = schedule["entries"]
        active = context.plan["optimizer_smoke" if context.phase in {"smoke", "performance_benchmark"} else "training"]
        if schedule["status"] != "sealed" or digest(entries) != schedule["entries_sha256"] or schedule["updates"] != active["steps"] or active["batch_size"] != 4 or len(entries) != active["steps"] * 4:
            raise ValueError("Active native budget differs from sealed paired schedule")
        for n, row in enumerate(entries):
            if row["update"] != n // 4 + 1 or row["slot"] != n % 4 or (n % 4 < 3 and row["source"] != "D"):
                raise ValueError("Paired schedule no longer has common 3D+1 slots")
        inventory = _verified_json(path(p["auxiliary_inventory"]), p["auxiliary_inventory_sha256"])
        if inventory.get("hidden_test_loaded") is not False or inventory.get("normalization_source") != "D_train_only":
            raise ValueError("Auxiliary inventory crosses data or normalization boundary")
        auxiliary_root = path(p["auxiliary_root"])
        aux = {}
        branch_labels = {}
        pair_samplers = {}
        for item in inventory.get("inputs", []):
            key = (item["source"], item["episode"], item["frame"])
            labels_path = auxiliary_root / item["labels"]
            if not labels_path.resolve().is_relative_to(auxiliary_root.resolve()):
                raise ValueError("Auxiliary labels escaped collection root")
            label_identity = (labels_path, item["labels_sha256"])
            if label_identity not in branch_labels:
                branch_labels[label_identity] = _verified_json(labels_path, item["labels_sha256"])
            labels = branch_labels[label_identity]
            start = item["frame"]
            native_starts = labels.get("native_valid_chunk_starts", labels["valid_chunk_starts"])
            if native_starts != labels["valid_chunk_starts"]:
                raise ValueError("Native chunk inventory differs from true label validity")
            if "pair_sampler" in item:
                sampler_path = auxiliary_root / item["pair_sampler"]
                if not sampler_path.resolve().is_relative_to(auxiliary_root.resolve()):
                    raise ValueError("Paired sampler escaped collection root")
                sampler_identity = (sampler_path, item["pair_sampler_sha256"])
                if sampler_identity not in pair_samplers:
                    pair_samplers[sampler_identity] = _verified_json(sampler_path, item["pair_sampler_sha256"])
                sampler = pair_samplers[sampler_identity]
                grid = sampler["common_valid_chunk_starts"]
                if (digest(grid) != item["common_grid_sha256"] or digest(grid) != sampler["common_starts_sha256"]
                        or sampler["sufficient_common_starts"] is not True or sampler["pre_deviation_prefix_exact"] is not True
                        or sampler["raw_labels_changed"] is not False or len(grid) < sampler["minimum_common_chunk_starts"]
                        or grid != sorted(set(grid)) or start not in grid or set(grid) - set(native_starts)):
                    raise ValueError("Auxiliary input is outside audited common native chunk grid")
                role = item["source"]
                if sampler[role+"_native_valid_chunk_starts"] != native_starts:
                    raise ValueError("Common-grid sampler differs from native branch labels")
            rows = labels["rows"][start:start + 50]
            if start not in native_starts or len(rows) != 50 or any(r["positive_target"] is not True or not r["safe"] for r in rows) or [r["step"] for r in rows] != list(range(start, start + 50)):
                raise ValueError("Auxiliary chunk includes unknown/deviation/unsafe target")
            if key in aux:
                raise ValueError("Repeated auxiliary native input identity")
            aux[key] = rows
        # Pinned upstream calls torch.utils.data.DataLoader directly. Keep the
        # test facade compatible, and bind/restore the actual callable owner.
        self.original_torch = None
        if hasattr(module,"DataLoader"):
            self.loader_owner = module
        else:
            # A global replacement breaks Accelerate's isinstance checks.
            # Override this native module's access path while retaining the
            # actual global DataLoader class and its objects unchanged.
            class Proxy:
                def __init__(self,original,**overrides):
                    self.original,self.overrides=original,overrides
                def __getattr__(self,name):
                    return self.overrides[name] if name in self.overrides else getattr(self.original,name)
            self.original_torch=module.torch
            self.loader_owner=Proxy(module.torch.utils.data,DataLoader=module.torch.utils.data.DataLoader)
            self.torch_wrapper=Proxy(module.torch,utils=Proxy(module.torch.utils,data=self.loader_owner))
        original = self.loader_owner.DataLoader
        if getattr(original, "_rosetta_paired_mixture", False):
            raise RuntimeError("Paired DataLoader already installed")
        self.original_policy = module.make_policy
        self.original_optimizer_factory=getattr(module,"make_optimizer_and_scheduler",None)
        if self.original_optimizer_factory is not None:
            def optimizer_factory(cfg,policy):
                if cfg.steps!=active["steps"]:raise ValueError("Native optimizer stop differs from sealed phase")
                scheduled=scheduler_horizon_config(cfg,context.plan["training"]["steps"])
                optimizer,scheduler=self.original_optimizer_factory(scheduled,policy)
                from rosetta_reality.vla.training.paired_mixture import save_bundle
                import dataclasses
                save_bundle(context.plan_path.parent/(context.run_name+"-actual-optimizer.json"),{
                    "actual_training_stop_updates":cfg.steps,"scheduler_horizon_updates":scheduled.steps,
                    "optimizer_configuration":dataclasses.asdict(cfg.optimizer),
                    "scheduler_configuration":dataclasses.asdict(cfg.scheduler),
                    "initial_learning_rates":[g["lr"] for g in optimizer.param_groups],
                    "scheduler_initial_state":scheduler.state_dict() if scheduler is not None else None,
                    "native_loss_changed":False})
                return optimizer,scheduler
            self.optimizer_wrapper=optimizer_factory
        self.observed_policies = []
        def make_policy(*args, **kwargs):
            policy = self.original_policy(*args, **kwargs)
            cuda_budget()
            from rosetta_reality.vla.training.paired_mixture import save_bundle
            tensors = {}
            for name, tensor in policy.state_dict().items():
                raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                tensors[name] = {"sha256":hashlib.sha256(raw).hexdigest(),"dtype":str(tensor.dtype),"shape":list(tensor.shape)}
            save_bundle(context.plan_path.parent / (context.run_name + "-actual-initialization.json"), {"status":"captured","tensors":tensors,"identity_sha256":digest(tensors),"optimizer_state_reused":False})
            if getattr(context,"device",None)=="cuda":torch.cuda.reset_peak_memory_stats()
            self.observed_policies.append(policy)
            return policy
        self.policy_wrapper = make_policy

        class MixtureDataset:
            def __init__(self, dataset):
                self.original = dataset
                episode = dataset.hf_dataset["episode_index"]
                frame = dataset.hf_dataset["frame_index"]
                allowed = set(context.experiment["dataset"]["train_episodes"])
                if set(int(x) for x in episode) - allowed:
                    raise ValueError("D native view includes protected episodes")
                self.d_indices = {(int(ep), int(fr)): i for i, (ep, fr) in enumerate(zip(episode, frame, strict=True))}
                self.template = dataset[0]
                self.identities = entries

            def __len__(self):
                return len(self.identities)

            def __getattr__(self, name):
                return getattr(self.original, name)

            def __getitem__(self, index):
                identity = self.identities[index]
                if identity["source"] == "D":
                    return self.original[self.d_indices[(identity["episode"], identity["frame"])]]
                rows = aux[(identity["source"], identity["episode"], identity["frame"])]
                first = rows[0]
                image_path = auxiliary_root / first["images"]
                if not image_path.resolve().is_relative_to(auxiliary_root.resolve()) or hashlib.sha256(image_path.read_bytes()).hexdigest() != first["images_sha256"]:
                    raise ValueError("Auxiliary image identity changed")
                with np.load(image_path, allow_pickle=False) as images:
                    output = {}
                    for key, value in self.template.items():
                        if key in p["camera_map"]:
                            pixels = images[p["camera_map"][key]]
                            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                                raise ValueError("Auxiliary camera is not native uint8 HWC RGB")
                            tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
                            output[key] = tensor if value.dtype == torch.uint8 else tensor.to(value.dtype).div(255)
                            if output[key].shape != value.shape:
                                raise ValueError("Auxiliary camera shape differs from D")
                        elif key == "observation.state":
                            output[key] = torch.as_tensor(first["state"], dtype=value.dtype)
                        elif key == "action":
                            output[key] = torch.as_tensor([r["action"] for r in rows], dtype=value.dtype)
                        elif key == "action_is_pad":
                            output[key] = torch.zeros_like(value, dtype=torch.bool)
                        elif key == "task":
                            output[key] = first["task"]
                        elif key in {"episode_index", "frame_index", "index", "timestamp", "next.done", "task_index"}:
                            output[key] = torch.zeros_like(value) if torch.is_tensor(value) else value
                        else:
                            raise ValueError("Unregistered native batch key: " + key)
                return output

        class Batches(torch.utils.data.BatchSampler):
            def __init__(self):
                # Preserve the sealed sequential index order while exposing
                # the native BatchSampler contract consumed by Accelerate.
                super().__init__(torch.utils.data.SequentialSampler(range(len(entries))),4,True)

        def loader(dataset, *args, **kwargs):
            if args or kwargs.get("batch_size") != 4 or kwargs.get("num_workers", 0) != 0:
                raise ValueError("Paired native loader requires batch4/workers0/no positional overrides")
            for key in ("batch_size", "shuffle", "sampler", "drop_last"):
                kwargs.pop(key, None)
            if "batch_sampler" in kwargs:
                raise ValueError("Unexpected second batch sampler")
            return original(MixtureDataset(dataset), batch_sampler=Batches(), **kwargs)
        loader._rosetta_paired_mixture = True
        self.original, self.wrapper = original, loader
        self.original_update = module.update_policy
        self.completed_updates = 0
        self.update_measurements=[]
        self.measured_target_batches = 0
        self.valid_target_slots = {"common_D":0,"auxiliary":0}
        self.padding_target_slots = {"common_D":0,"auxiliary":0}
        import functools
        import inspect
        signature = inspect.signature(self.original_update)
        @functools.wraps(self.original_update)
        def update(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            batch = bound.arguments.get("batch")
            measured = None
            if batch is not None:
                mask = batch.get("action_is_pad")
                if not torch.is_tensor(mask) or mask.dtype != torch.bool or tuple(mask.shape) != (4,50):
                    raise ValueError("Native paired target-mask contract differs before optimizer")
                measured = {"common_D":int(mask[:3].sum().item()),"auxiliary":int(mask[3:].sum().item())}
            # Accelerator may install its own autocast forward after policy
            # construction. Observe the live callable only for this update,
            # preserving that upstream wrapper exactly on exit.
            policy = bound.arguments.get("policy")
            original_forward = policy.forward if policy is not None else None
            def finite_forward(*forward_args, **forward_kwargs):
                value = original_forward(*forward_args, **forward_kwargs)
                loss = value[0] if isinstance(value, tuple) else value.get("loss") if isinstance(value, dict) else value
                if not torch.is_tensor(loss) or loss.numel()!=1 or not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("Paired native loss is nonfinite or not scalar before backward")
                cuda_budget()
                return value
            if policy is not None:policy.forward = finite_forward
            import time
            started=time.monotonic()
            try:result = self.original_update(*args, **kwargs)
            finally:
                if policy is not None:
                    if policy.forward is not finite_forward:
                        raise RuntimeError("Native forward changed during a registered update")
                    policy.forward = original_forward
            accelerator = bound.arguments.get("accelerator")
            if accelerator is not None and getattr(accelerator, "optimizer_step_was_skipped", False):
                raise RuntimeError("Paired mixture optimizer update was skipped")
            self.completed_updates += 1
            import resource
            optimizer=bound.arguments.get("optimizer")
            self.update_measurements.append({"successful_update":self.completed_updates,
                "native_update_call_seconds":time.monotonic()-started,
                "learning_rates_after_update":[g["lr"] for g in optimizer.param_groups] if optimizer is not None else [],
                "process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                "cuda_peak_allocated":torch.cuda.max_memory_allocated() if getattr(context,"device",None)=="cuda" else None,
                "cuda_peak_reserved":torch.cuda.max_memory_reserved() if getattr(context,"device",None)=="cuda" else None})
            if measured is not None:
                self.measured_target_batches += 1
                for name,total in (("common_D",150),("auxiliary",50)):
                    self.padding_target_slots[name] += measured[name]
                    self.valid_target_slots[name] += total-measured[name]
            if self.completed_updates > schedule["updates"]:
                raise RuntimeError("Actual optimizer updates exceed paired budget")
            cuda_budget()
            return result
        self.update_wrapper = update
        module.update_policy = update
        if self.original_torch is not None:
            self.loader_owner.overrides["DataLoader"]=loader
            module.torch=self.torch_wrapper
        else:self.loader_owner.DataLoader = loader
        module.make_policy = make_policy
        if self.original_optimizer_factory is not None:module.make_optimizer_and_scheduler=self.optimizer_wrapper

    def restore(self, context):
        import lerobot.scripts.lerobot_train as module
        if self.loader_owner.DataLoader is not self.wrapper:
            raise RuntimeError("Paired DataLoader changed before restore")
        if self.original_torch is not None:
            if module.torch is not self.torch_wrapper:raise RuntimeError("Native torch access binding changed before restore")
            module.torch=self.original_torch
        else:self.loader_owner.DataLoader = self.original
        if module.update_policy is not self.update_wrapper:
            raise RuntimeError("Paired update observer changed before restore")
        module.update_policy = self.original_update
        if module.make_policy is not self.policy_wrapper:
            raise RuntimeError("Paired policy initialization observer changed before restore")
        module.make_policy = self.original_policy
        if self.original_optimizer_factory is not None:
            if module.make_optimizer_and_scheduler is not self.optimizer_wrapper:
                raise RuntimeError("Native optimizer constructor changed before restore")
            module.make_optimizer_and_scheduler=self.original_optimizer_factory
        if self.observed_policies:
            import torch
            from rosetta_reality.vla.training.paired_mixture import digest,save_bundle
            tensors={}
            for name,tensor in self.observed_policies[-1].state_dict().items():
                raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                tensors[name]={"sha256":hashlib.sha256(raw).hexdigest(),"dtype":str(tensor.dtype),"shape":list(tensor.shape)}
            save_bundle(context.plan_path.parent/(context.run_name+"-actual-final-parameters.json"),
                {"identity_sha256":digest(tensors),"tensors":tensors,"successful_optimizer_updates":self.completed_updates})
        # This records actual completed native calls, never planned exposure as execution.
        from rosetta_reality.vla.training.paired_mixture import save_bundle
        report = context.plan_path.parent / (context.run_name + "-actual-updates.json")
        expected_updates = 0 if context.phase == "performance_benchmark" else context.plan["optimizer_smoke" if context.phase == "smoke" else "training"]["steps"]
        save_bundle(report, {"status":"complete" if self.completed_updates == expected_updates else "incomplete", "successful_optimizer_updates":self.completed_updates,"sample_exposures":self.completed_updates*4,"D_common_exposures":self.completed_updates*3,"auxiliary_exposures":self.completed_updates,"native_loss_changed":False,"phase":context.phase,"target_mask_batches_measured":self.measured_target_batches,"valid_target_slots_actual":self.valid_target_slots,"padding_target_slots_actual":self.padding_target_slots,"equal_effective_targets_across_three_arms_claimed":False,"update_measurements":self.update_measurements,"update_timing_excludes_native_data_loading_and_processor":True})
