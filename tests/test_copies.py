"""The code copied from linker-lab is unchanged (opt-in: needs a linker-lab checkout).

    LINKER_LAB_CHECKOUT=~/linker-lab uv run pytest tests/test_copies.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKOUT = os.environ.get("LINKER_LAB_CHECKOUT")
TOOL = Path(__file__).parents[1] / "tools" / "check_copies.py"


@pytest.mark.skipif(not CHECKOUT, reason="set LINKER_LAB_CHECKOUT to a linker-lab checkout")
def test_copies_match_linker_lab():
    result = subprocess.run([sys.executable, str(TOOL), CHECKOUT], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
