import sys
from pathlib import Path


def ensure_runtime_paths() -> Path:
    root = Path(__file__).resolve().parents[3]
    dust3r_root = root / "submodules" / "dust3r"
    croco_root = dust3r_root / "croco"

    for path in (dust3r_root, croco_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root
