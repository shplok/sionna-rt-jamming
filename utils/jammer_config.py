"""Persist jammer initial positions into main.py."""
import re
from pathlib import Path

import numpy as np


def update_initial_position_in_main(
    main_py_path: str,
    jammer_name: str,
    position: np.ndarray,
    use_z_height_symbol: bool = True,
) -> bool:
    """
    Update one jammer's initial_position line inside initial_jammers_config in main.py.
    Returns True if the file was updated.
    """
    path = Path(main_py_path)
    lines = path.read_text().splitlines()
    in_target = False

    for i, line in enumerate(lines):
        if f'"name": "{jammer_name}"' in line:
            in_target = True
            continue

        if not in_target:
            continue

        if '"name":' in line and jammer_name not in line:
            break

        if re.search(r'"(?:initial_)?position":\s*np\.array\(', line):
            x, y, z = float(position[0]), float(position[1]), float(position[2])
            if use_z_height_symbol and abs(z - 1.5) < 1e-6:
                z_expr = "Z_HEIGHT"
            else:
                z_expr = f"{z:.1f}"
            indent = line[: len(line) - len(line.lstrip())]
            lines[i] = (
                f'{indent}"initial_position": np.array([{x:.1f}, {y:.1f}, {z_expr}]),'
            )
            path.write_text("\n".join(lines) + "\n")
            return True

    return False
