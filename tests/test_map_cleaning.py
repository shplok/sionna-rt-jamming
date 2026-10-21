"""Tests for utils/map_cleaning.py.  CPU only, no Sionna.

    python -m pytest tests/test_map_cleaning.py      # or
    python tests/test_map_cleaning.py

The regression test reproduces stored cleaned maps from their stored raw per-seed maps,
bit for bit. It runs when the cleaned dataset is found (RTFIX_DATASETS, default
./datasets_rtfix) and is skipped otherwise.
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils import map_cleaning as mc  # noqa: E402

N, CELL, B = 300, 10.0, 1500.0
CENTER, SIZE = [0.0, 0.0, 1.5], (2 * B, 2 * B)
XX, YY = mc.cell_centres(CENTER, SIZE, (CELL, CELL), (N, N))
CFG = mc.make_config([42, 7, 123], cap_w=10.0, noise_floor_w=8e-15)
TX = (5.0, -5.0)                       # jammer in cell (row 149, col 150)


def smooth_map():
    """1/r^2-like field, ~1e-6 W at 1 km; no spikes."""
    r = np.hypot(XX - TX[0], YY - TX[1]) + 5.0
    return 1e-1 / r ** 2


def three(m):
    return np.stack([m, m, m])


def rc(x, y):
    return int((y + B) // CELL), int((x + B) // CELL)


def test_cell_centres_match_sionna_layout():
    assert XX.shape == (N, N) and XX[0, 0] == -1495.0 and YY[0, 0] == -1495.0
    assert XX[0, 1] == -1485.0 and YY[1, 0] == -1485.0       # x along columns, y along rows


def test_smooth_map_untouched():
    m = smooth_map()
    w, masks = mc.clean(three(m), TX, XX, YY, CFG)
    assert not masks["touched"].any()
    np.testing.assert_array_equal(w, m)


def test_median_rejects_single_seed_spike():
    m = smooth_map()
    bad = m.copy(); r, c = rc(600, 600); bad[r, c] = 1e3
    w, masks = mc.clean(np.stack([m, bad, m]), TX, XX, YY, CFG)
    assert not masks["touched"].any() and w[r, c] == m[r, c]


def test_despike_far_from_jammer():
    m = smooth_map(); r, c = rc(600, 600); m[r, c] = 1e-2     # ~+40 dB over its neighbours
    w, masks = mc.clean(three(m), TX, XX, YY, CFG)
    assert masks["despike"][r, c] and masks["despike"].sum() == 1
    nb = smooth_map()[r - 1:r + 2, c - 1:c + 2].ravel()
    assert np.isclose(w[r, c], np.median(np.delete(nb, 4)))


def test_no_despike_near_jammer_or_below_floor():
    m = smooth_map()
    r1, c1 = rc(TX[0] + 20, TX[1]); m[r1, c1] *= 1e4         # 20 m from the jammer
    r2, c2 = rc(-1400, 1400); m[r2, c2] = 1e-11               # -110 dBW, below the floor
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if (i, j) != (0, 0):
                m[r2 + i, c2 + j] = 1e-15
    w, masks = mc.clean(three(m), TX, XX, YY, CFG)
    assert not masks["despike"][r1, c1] and not masks["despike"][r2, c2]


def test_cap_at_transmit_power():
    m = smooth_map(); r, c = rc(TX[0], TX[1])
    m[r - 1:r + 2, c - 1:c + 2] = 50.0                        # clump above 10 W at the jammer
    w, masks = mc.clean(three(m), TX, XX, YY, CFG)
    assert w.max() <= 10.0 and masks["cap"][r, c]


def test_nonfinite_repaired():
    m = smooth_map(); r, c = rc(-800, 300); m[r, c] = np.nan
    w, masks = mc.clean(three(m), TX, XX, YY, CFG)
    assert masks["nonfinite"][r, c] and np.isfinite(w).all() and w[r, c] > 0


def test_record_guard():
    with tempfile.TemporaryDirectory() as d:
        mc.ensure_record(d, CFG, has_maps=False)
        mc.ensure_record(d, dict(CFG), has_maps=True)         # same settings: fine
        other = mc.make_config([42, 7, 123], cap_w=10.0, noise_floor_w=8e-15, despike_db=10)
        for cfg in (other, mc.NO_CLEANING):
            try:
                mc.ensure_record(d, cfg, has_maps=True)
            except SystemExit:
                pass
            else:
                raise AssertionError("mixed cleaning settings were accepted")
    with tempfile.TemporaryDirectory() as d:                  # maps without a record
        try:
            mc.ensure_record(d, CFG, has_maps=True)
        except SystemExit:
            pass
        else:
            raise AssertionError("cleaning into an uncleaned folder was accepted")
        rec = mc.ensure_record(d, dict(mc.NO_CLEANING), has_maps=True)
        assert rec["method"] == "none" and "note" in rec
        with open(os.path.join(d, mc.RECORD_NAME)) as f:
            assert json.load(f)["method"] == "none"


def test_matches_rtfix_dc():
    """Same output as the agent's utils/rtfix_dc.py (srt-rtfix), if that copy is around."""
    src = os.environ.get("RTFIX_DC_DIR", os.path.join(os.path.dirname(ROOT), "srt-rtfix"))
    if not os.path.exists(os.path.join(src, "utils", "rtfix_dc.py")):
        print("  skip: rtfix_dc.py not found"); return
    import importlib.util
    spec = importlib.util.spec_from_file_location("rtfix_dc", os.path.join(src, "utils", "rtfix_dc.py"))
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    rng = np.random.default_rng(0)
    for _ in range(5):
        base = smooth_map()
        maps = base * rng.lognormal(0, 0.5, size=(3, N, N))
        idx = rng.integers(0, N, size=(3, 60, 2))
        for s in range(3):                                    # spikes, hot cells, NaNs
            maps[s, idx[s, :40, 0], idx[s, :40, 1]] = 10 ** rng.uniform(-6, 2, 40)
            maps[s, idx[s, 40:, 0], idx[s, 40:, 1]] = np.nan
        tx = rng.uniform(-1400, 1400, 2)
        w1, m1 = mc.clean(maps, tx, XX, YY, CFG)
        w2, m2 = ref.clean(maps, tx)
        np.testing.assert_array_equal(w1, w2)
        for k in m1:
            np.testing.assert_array_equal(m1[k], m2[k])


def test_reproduces_stored_rtfix_maps(n_static=40):
    """Raw per-seed maps -> cleaned map, compared with what is on disk (exact)."""
    ds = os.environ.get("RTFIX_DATASETS", os.path.join(ROOT, "datasets_rtfix"))
    st = os.path.join(ds, "batch_simulation_nyc", "single_static_jammers")
    raw = os.path.join(ds, "_raw", "batch_simulation_nyc", "single_static_jammers")
    if not os.path.exists(os.path.join(raw, "watts_s42.npy")):
        print("  skip: no stored rtfix maps"); return
    pos = np.load(os.path.join(st, "positions.npy"))
    stored = np.load(os.path.join(st, "watts.npy"), mmap_mode="r")
    counts = np.load(os.path.join(raw, "clean_counts.npy"))
    seeds = [np.load(os.path.join(raw, f"watts_s{s}.npy"), mmap_mode="r") for s in (42, 7, 123)]
    # every position with a despiked or capped cell, then evenly spaced others
    ids = np.flatnonzero(counts[:, 0] + counts[:, 1] > 0)[:n_static // 2]
    ids = np.unique(np.r_[ids, np.linspace(0, len(pos) - 1, n_static // 2).astype(int)])
    for i in ids:
        w, masks = mc.clean(np.stack([s[i] for s in seeds]), pos[i][:2], XX, YY, CFG)
        np.testing.assert_array_equal(w.astype(np.float32), stored[i], err_msg=f"static {i}")
        assert mc.mask_counts(masks) == counts[i].tolist(), f"static {i} counts"

    tr = os.path.join(ds, "batch_simulation_nyc", "single_trajectory_jammers")
    rawt = os.path.join(ds, "_raw", "batch_simulation_nyc", "single_trajectory_jammers")
    tid = "traj_dur30s_vel03mps_00"
    if os.path.exists(os.path.join(rawt, f"rm_watts_raw_{tid}.npy")):
        path = np.load(os.path.join(tr, f"{tid}.npy"))
        r = np.load(os.path.join(rawt, f"rm_watts_raw_{tid}.npy"), mmap_mode="r")   # (S, T, H, W)
        st_w = np.load(os.path.join(tr, "radio_maps", f"rm_watts_{tid}.npy"), mmap_mode="r")
        for t in range(0, r.shape[1], 7):
            w, _ = mc.clean(r[:, t], path[t][:2], XX, YY, CFG)
            np.testing.assert_array_equal(w.astype(np.float32), st_w[t], err_msg=f"{tid} t={t}")
    print(f"  {len(ids)} static maps (of which {int(((counts[ids, 0] + counts[ids, 1]) > 0).sum())} "
          f"with despiked/capped cells) reproduced exactly")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
