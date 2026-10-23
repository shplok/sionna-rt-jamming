#!/bin/bash
# ==============================================================================
# 20-position GPU smoke test for simulate_static.  [GPU]
#
# A standalone script rather than an inline `bash -c` string, because the inline
# version could not reliably activate the environment on the remote node: it was
# written before the project moved to a venv and still called `conda activate`,
# and it did not forward DRJIT_LIBLLVM_PATH, without which `import sionna.rt`
# fails outright on Explorer.
#
# Sourcing server_env.sh here means the remote node derives everything itself
# (venv, libLLVM, cache redirects) instead of inheriting a half-built environment.
#
# Run it directly on a GPU node, or let ./submit.sh smoke dispatch it via srun.
# ==============================================================================
set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DS="${SMOKE_DATASET_DIR:-./datasets/batch_simulation_smoke}"
N="${SMOKE_N:-20}"

source ./server_env.sh
sionna_activate_env || exit 1

echo "host        : $(hostname)"
echo "python      : $(command -v python)  ($(python -V 2>&1))"
echo "libLLVM     : ${DRJIT_LIBLLVM_PATH:-<unset>}"
echo "variant     : $MITSUBA_VARIANT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || { echo "ERROR: no GPU visible on this node"; exit 1; }
echo

rm -rf "$DS"

echo "--- generate_static ($N positions) ---"
python main_batch_cluster.py --action generate_static --dataset-dir "$DS" \
    --n-static "$N" --map-bounds-b 1500 --cell-size 10 10

echo "--- simulate_static [GPU] ---"
python main_batch_cluster.py --action simulate_static --dataset-dir "$DS" \
    --map-bounds-b 1500 --cell-size 10 10 --samples-per-tx 10000000 --max-depth 80

echo "--- checking output ---"
SMOKE_DS="$DS" SMOKE_N="$N" python - <<'PY'
import os, numpy as np
ds, n = os.environ["SMOKE_DS"], int(os.environ["SMOKE_N"])
a = np.load(f"{ds}/single_static_jammers/watts.npy", mmap_mode="r")
p = np.load(f"{ds}/single_static_jammers/positions.npy")
print("  watts.npy    ", a.shape, a.dtype)
print("  positions.npy", p.shape, p.dtype)
assert a.shape == (n, 300, 300), f"expected ({n}, 300, 300), got {a.shape}"
assert a.dtype == np.float32, a.dtype
m = float(np.asarray(a[0], dtype=np.float32).max())
print(f"  max watts in map 0: {m:.6g}")
assert m > 0, "map is all zeros - the solver returned nothing"
finite = np.isfinite(np.asarray(a[0], dtype=np.float32))
print(f"  finite cells in map 0: {100*finite.mean():.2f}%")
print("SMOKE TEST PASSED")
PY

echo
echo "Remove the scratch dataset with:  rm -rf $DS"
