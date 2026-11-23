"""Pure label contract negatives; actual D40/processor integration runs separately."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("cpu_real",Path(__file__).resolve().parents[1]/"scripts/check_paired_real_cpu_inputs.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [{"step":i,"safe":True,"positive_target":True,"action_source":"state_conditioned_teacher"} for i in range(50)]


def test_complete_actual_targets_contract():module.verify_chunk_rows(rows(),0)


@pytest.mark.parametrize("kind",["short","gap","unsafe","deliberate","unqualified"])
def test_bad_target_chunks_rejected(kind):
    values=rows()
    if kind=="short":values.pop()
    elif kind=="gap":values[1]["step"]=2
    elif kind=="unsafe":values[1]["safe"]=False
    elif kind=="deliberate":values[1]["action_source"]="registered_deliberate_deviation"
    else:values[1]["positive_target"]=False
    with pytest.raises(ValueError):module.verify_chunk_rows(values,0)
