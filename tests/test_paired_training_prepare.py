"""Synthetic preparation to the real formal launcher split validator; no ML load."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inputs(tmp_path, *, omit=False, extra=False):
    from rosetta_reality.vla import load_smolvla_experiment
    experiment = load_smolvla_experiment(ROOT / "configs/vla/smolvla_450m_aloha_insertion_action_repair_bounded_gripper_003.yaml", ROOT)
    episodes = list(experiment["dataset"]["train_episodes"])
    selected = episodes[:-1] if omit else episodes + [999] if extra else episodes
    original = tmp_path / "original.json"
    original.write_text(json.dumps({"source_split":"train","hidden_test_loaded":False,"inputs":[
        {"episode":ep,"frame":0,"positive_supervision":True,"chunk_crosses_unqualified_action":False}
        for ep in selected]}))
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"pairs":[],"runtime_inputs":[],"camera_map":{}}))
    return experiment, original, cohort


def test_prepare_to_formal_validation_preserves_exact_parent_order(tmp_path):
    import yaml
    prepare = module("paired_prepare_test", ROOT / "scripts/prepare_paired_training_bundle.py")
    launcher = module("paired_launcher_test", ROOT / "scripts/run_smolvla_v2.py")
    experiment,original,cohort = inputs(tmp_path)
    output = tmp_path / "bundle"
    prepare.prepare(Namespace(template=str(ROOT/"configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml"),
                              original=str(original),cohort=str(cohort),output=str(output),auxiliary_root="collection"))
    formal = yaml.safe_load((output/"plan-A-train.yaml").read_text())
    assert formal["training"]["episodes"] == experiment["dataset"]["train_episodes"]
    assert formal["training"]["episodes"] != sorted(formal["training"]["episodes"])
    launcher._validate_split(formal, experiment, "train")
    smoke = yaml.safe_load((output/"plan-A-smoke.yaml").read_text())
    launcher._validate_split(smoke, experiment, "smoke")
    from rosetta_reality.vla.training.integrity import validate_local_implementation,CORE_IMPLEMENTATION
    for registered in (smoke,formal):
        checked=validate_local_implementation(registered,ROOT)
        assert CORE_IMPLEMENTATION <= checked.keys()
    from rosetta_reality.vla.training.launch import build_training_arguments,optimizer_arguments
    assert smoke["smoke_optimizer_configuration_source"]==formal["smoke_optimizer_configuration_source"]=="training.optimizer_and_scheduler"
    for key in ("optimizer","scheduler"):
        assert key not in smoke["optimizer_smoke"]
        assert formal["training"][key] == smoke["training"][key]
    for phase,registered in (("smoke",smoke),("train",formal)):
        actual=build_training_arguments(plan=registered,experiment=experiment,mode=phase,run_name="assembly-test",
            model_root=tmp_path/"model",dataset_root=tmp_path/"view",output_dir=tmp_path/"checkpoint",device="cuda")
        assert all(fragment in actual for fragment in optimizer_arguments(formal["training"]))
    paired_launcher=module("paired_semantics_test",ROOT/"scripts/run_paired_training_v2.py")
    assert paired_launcher.engineering_semantics(smoke,output/"plan-A-smoke.yaml",experiment) == paired_launcher.engineering_semantics(formal,output/"plan-A-train.yaml",experiment)
    assert json.loads((output/"bundle.json").read_text())["unavailable_arms"] == ["B","C"]


@pytest.mark.parametrize("omit,extra",[(True,False),(False,True)])
def test_inventory_incomplete_or_nonparent_is_rejected_before_output(tmp_path,omit,extra):
    prepare = module("paired_prepare_reject_test", ROOT / "scripts/prepare_paired_training_bundle.py")
    _,original,cohort = inputs(tmp_path,omit=omit,extra=extra)
    output = tmp_path/"bundle"
    with pytest.raises(ValueError,match="exactly every parent"):
        prepare.prepare(Namespace(template=str(ROOT/"configs/vla/m2-smolvla450m-visual-native-b4-pilot-003.yaml"),
                                  original=str(original),cohort=str(cohort),output=str(output),auxiliary_root="collection"))
    assert not output.exists()
