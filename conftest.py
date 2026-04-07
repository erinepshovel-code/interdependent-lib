"""
Root conftest.py.

If the `cryptography` package is not functional (e.g. broken system install),
skip test_pcea.py entirely rather than letting its collection panic pytest.
"""

import subprocess
import sys

def _cryptography_ok() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "from cryptography.hazmat.primitives import hashes"],
        capture_output=True,
    )
    return result.returncode == 0

if not _cryptography_ok():
    collect_ignore = ["tests/test_pcea.py"]
