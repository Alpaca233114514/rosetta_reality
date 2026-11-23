"""Pure accounting negatives; live provenance is independently audited on worker."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location("v3_audit",Path(__file__).resolve().parents[1]/"scripts/audit_paired_collection_v3.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_prefix_counted_once_physical_twice_logical():
    assert module.accounting(598,175,4) == 427


@pytest.mark.parametrize("values",[(-1,0,0),(1,2,0),(1,0,-1),(1.0,0,0),(True,0,0)])
def test_impossible_or_noninteger_accounting_is_rejected(values):
    with pytest.raises(ValueError):
        module.accounting(*values)
