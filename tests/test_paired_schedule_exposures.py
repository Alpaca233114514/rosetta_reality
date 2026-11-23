"""Metadata-only accounting negatives, never training-execution evidence."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("exposures",Path(__file__).resolve().parents[1]/"scripts/summarize_paired_schedule_exposures.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(source="D"):
    original={"hidden_test_loaded":False,"source_split":"train","inputs":[{"episode":1,"frame":499,"valid_target_slots":1,"padding_slots":49}]}
    aux={"R_inputs":[{"episode":1,"frame":499,"positive_supervision":True,"chunk_crosses_unqualified_action":False}]}
    entries=[{"update":1,"slot":slot,"source":"D" if slot<3 else source,"episode":1,"frame":499} for slot in range(4)]
    schedule={"status":"sealed","updates":1,"batch_size":4,"entries":entries,"entries_sha256":module.digest(entries)}
    return schedule,original,aux


def test_equal_inputs_unequal_valid_slots_are_reported():
    a=module.summarize(*fixture())
    b=module.summarize(*fixture("R"))
    assert a["planned_input_exposures"]==b["planned_input_exposures"]==4
    assert a["valid_target_slot_exposures"]==4 and b["valid_target_slot_exposures"]==53
    assert a["padding_slot_exposures"]==196 and b["padding_slot_exposures"]==147
    assert a["unique_input_frames"]==1 and a["repeated_input_exposures"]==3


def test_mutated_sealed_schedule_is_rejected():
    values=fixture()
    values[0]["entries"][0]["frame"]=498
    with pytest.raises(ValueError):module.summarize(*values)


def test_no_hidden_inventory_is_opened():
    values=fixture()
    values[1]["hidden_test_loaded"]=True
    with pytest.raises(ValueError):module.summarize(*values)
