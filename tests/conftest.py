# SPDX-License-Identifier: Apache-2.0
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "nkisim: needs NKI_SIMULATOR=1 (run in a separate pytest process)"
    )
