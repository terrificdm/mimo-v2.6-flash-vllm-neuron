# SPDX-License-Identifier: Apache-2.0
"""Import the (patched) HF MiMo-V2 remote code as a regular package for module-level tests."""

import importlib
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = "mimo_hf_ref"


def load():
    if _PKG in sys.modules:
        return sys.modules[_PKG + ".modeling_mimo_v2"], sys.modules[_PKG + ".configuration_mimo_v2"]
    d = tempfile.mkdtemp(prefix="mimo_hf_ref_")
    pkg = os.path.join(d, _PKG)
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    shutil.copy(os.path.join(ROOT, "reference/hf_patched/modeling_mimo_v2.py"), pkg)
    shutil.copy(os.path.join(ROOT, "reference/hf/configuration_mimo_v2.py"), pkg)
    sys.path.insert(0, d)
    cfg_mod = importlib.import_module(_PKG + ".configuration_mimo_v2")
    mdl_mod = importlib.import_module(_PKG + ".modeling_mimo_v2")
    return mdl_mod, cfg_mod
