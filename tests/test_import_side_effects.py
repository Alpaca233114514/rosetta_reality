"""Import boundaries must not activate optional data clients."""

import subprocess
import sys


def test_data_import_does_not_import_lerobot_or_huggingface_hub() -> None:
    code = (
        "import sys; import rosetta_reality.data; "
        "assert 'lerobot' not in sys.modules; "
        "assert 'huggingface_hub' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
