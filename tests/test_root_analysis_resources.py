import pytest

from scripts.root_analysis_resources import verify_resource_envelope


def test_actual_resource_limits_not_just_yaml(tmp_path):
    plan = {"resources": {"memory_mib": 1536, "cpu_threads": 2, "wall_seconds": 180}}
    (tmp_path / "memory.max").write_text(str(1536 * 1024**2))
    (tmp_path / "cpu.max").write_text("200000 100000")
    env = {"ROSETTA_ROOT_ENFORCED_WALL_SECONDS": "180"}
    assert verify_resource_envelope(plan, cgroup=tmp_path, environment=env)["cpu_quota"] == 2
    (tmp_path / "memory.max").write_text(str(3 * 1024**3))
    with pytest.raises(ValueError, match="memory exceeds"):
        verify_resource_envelope(plan, cgroup=tmp_path, environment=env)
    (tmp_path / "memory.max").write_text(str(1536 * 1024**2))
    with pytest.raises(ValueError, match="timeout absent"):
        verify_resource_envelope(plan, cgroup=tmp_path, environment={})
    (tmp_path / "cpu.max").write_text("max 100000")
    with pytest.raises(ValueError, match="Bounded"):
        verify_resource_envelope(plan, cgroup=tmp_path, environment=env)
