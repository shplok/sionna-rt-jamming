"""
Radio-map cleaning: per-cell median over several solver seeds, then local repair.

Why: Sionna RT's RadioMapSolver is a Monte-Carlo estimator. With the jammer and the
measurement plane at the same height (1.5 m), a diffracted ray that reaches a cell at a
grazing angle gets a huge integration weight (radio_map.py, _diffraction_integration_weight,
n.d -> 0), and one such ray turns a single cell into an isolated spike, sometimes far
above the transmit power. A different solver seed puts the spikes in different cells, so
the per-cell median of three seeds removes almost all of them; the few that survive are
repaired from their neighbours. Sionna itself is not patched.

Steps, for one single-jammer map (watts, shape (H, W)):
  1. per-cell median over the seeds;
  2. despike: a cell is a spike if it is more than `despike_db` above EVERY one of its 8
     neighbours, more than `despike_excl_m` from the jammer (the true peak is a legitimate
     isolated maximum) and above `despike_floor_dbw` (below it a spike cannot matter
     against the noise floor). Spikes get the median of their finite, positive neighbours;
  3. cap: cells above the transmit power `cap_w` get the neighbour median, up to
     `max_cap_passes` times; whatever is still above is clipped to `cap_w`;
  4. non-finite cells (NaN / inf in the median) get the neighbour median.

Known limitation: a clump of two or more adjacent spiky cells is not caught by step 2
(each cell has a high neighbour); step 3 still bounds it by the transmit power.

The defaults are the ones the datasets_rtfix maps were generated with; changing any of
them changes the dataset, and main_batch_cluster.py refuses to mix settings in one folder.
"""

import json
import os
import time
import warnings

import numpy as np

# Bump when the algorithm changes in a way that changes the output for the same settings.
CLEANING_VERSION = 1

DEFAULTS = {
    "despike_db": 20.0,
    "despike_floor_dbw": -100.0,
    "despike_excl_m": 30.0,
    "max_cap_passes": 5,
}


def make_config(seeds, cap_w, noise_floor_w, despike_db=DEFAULTS["despike_db"],
                despike_floor_dbw=DEFAULTS["despike_floor_dbw"],
                despike_excl_m=DEFAULTS["despike_excl_m"],
                max_cap_passes=DEFAULTS["max_cap_passes"]):
    """The settings that determine the cleaned maps (also what map_cleaning.json records)."""
    return {
        "method": "median_despike_cap",
        "cleaning_version": CLEANING_VERSION,
        "seeds": [int(s) for s in seeds],
        "despike_db": float(despike_db),
        "despike_floor_dbw": float(despike_floor_dbw),
        "despike_excl_m": float(despike_excl_m),
        "cap_w": float(cap_w),
        "noise_floor_w": float(noise_floor_w),
        "max_cap_passes": int(max_cap_passes),
    }


# The record written when cleaning is off: maps straight from the solver, default seed.
NO_CLEANING = {"method": "none"}


def cell_centres(map_center, map_size, cell_size, shape):
    """(X, Y) cell-centre coordinates in metres, shape (H, W), rows along y.

    Same formula as PlanarRadioMap.cell_centers (orientation 0), but in float64 from the
    arguments given to the solver, so the jammer-exclusion test is exactly reproducible.
    """
    h, w = shape
    xs = map_center[0] - 0.5 * map_size[0] + (np.arange(w) + 0.5) * cell_size[0]
    ys = map_center[1] - 0.5 * map_size[1] + (np.arange(h) + 0.5) * cell_size[1]
    return np.meshgrid(xs, ys)


def _neighbours(a):
    """The 8 neighbours of every cell, shape (8, H, W); NaN outside the map."""
    pad = np.pad(a, 1, constant_values=np.nan)
    h, w = a.shape
    return np.stack([pad[1 + i:h + 1 + i, 1 + j:w + 1 + j]
                     for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])


def _nb_median(w, mask):
    """Median of the finite, positive neighbours of the cells in `mask` (0 if there are none)."""
    nb = _neighbours(np.where(np.isfinite(w) & (w > 0), w, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods
        med = np.nanmedian(nb[:, mask], axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def spike_mask(w, tx_xy, xx, yy, cfg):
    """Isolated cells more than despike_db above all 8 neighbours (see module docstring)."""
    w0 = np.where(np.isfinite(w), w, 0.0)
    y = 10.0 * np.log10(w0 + cfg["noise_floor_w"])
    nbmax = np.nanmax(_neighbours(y), axis=0)
    far = np.hypot(xx - tx_xy[0], yy - tx_xy[1]) > cfg["despike_excl_m"]
    return (y - nbmax > cfg["despike_db"]) & far & (y > cfg["despike_floor_dbw"])


def clean(maps, tx_xy, xx, yy, cfg):
    """Clean one single-jammer map from its per-seed realisations.

    maps   (S, H, W) watts, one per seed
    tx_xy  jammer (x, y) in metres
    xx, yy cell centres from cell_centres()
    Returns (watts (H, W) float64, masks) with boolean masks despike / cap / nonfinite /
    touched (their union).
    """
    m = np.median(np.asarray(maps, np.float64), axis=0)
    nonf = ~np.isfinite(m)
    w = np.where(nonf, 0.0, m)

    sp = spike_mask(w, tx_xy, xx, yy, cfg)
    if sp.any():
        w[sp] = _nb_median(w, sp)

    cap = np.zeros_like(sp)
    for _ in range(cfg["max_cap_passes"]):
        hot = w > cfg["cap_w"]
        if not hot.any():
            break
        cap |= hot
        w[hot] = _nb_median(w, hot)
    w[w > cfg["cap_w"]] = cfg["cap_w"]

    if nonf.any():
        w[nonf] = _nb_median(np.where(nonf, np.nan, w), nonf)
    return w, dict(despike=sp, cap=cap, nonfinite=nonf, touched=sp | cap | nonf)


def mask_counts(masks):
    """[despike, cap, nonfinite, touched] cell counts, the row stored in clean_counts*.npy."""
    return [int(masks[k].sum()) for k in ("despike", "cap", "nonfinite", "touched")]


# -----------------------------------------------------------------------------
# Provenance: one map_cleaning.json per output folder
# -----------------------------------------------------------------------------
RECORD_NAME = "map_cleaning.json"


def _sionna_version():
    try:
        from importlib.metadata import version
        return version("sionna-rt")
    except Exception:
        return "unknown"


def read_record(folder):
    path = os.path.join(folder, RECORD_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def settings_of(record):
    """The part of a record that determines the maps (drops version / date / notes)."""
    return {k: v for k, v in record.items() if k not in ("sionna_version", "created", "note")}


def ensure_record(folder, cfg, has_maps, create=True):
    """Make sure `folder` records the cleaning settings `cfg`, and that they match.

    - record exists and matches: fine;
    - record exists and differs: refuse (the folder would mix differently cleaned maps);
    - no record, no maps yet: write it (only if `create`);
    - no record but maps already there (made before this record existed, i.e. without
      cleaning): fine if `cfg` is NO_CLEANING (recorded as such, with a note), refuse
      otherwise.
    """
    rec = read_record(folder)
    if rec is not None:
        if settings_of(rec) != settings_of(cfg):
            raise SystemExit(
                f"REFUSING: {os.path.join(folder, RECORD_NAME)} records the cleaning\n"
                f"  {settings_of(rec)}\nbut this run asks for\n  {settings_of(cfg)}\n"
                "Mixing the two in one folder would give an inconsistent dataset. Use the "
                "recorded settings, or a new --dataset-dir.")
        return rec
    if not create:
        raise SystemExit(f"REFUSING: no {RECORD_NAME} in {folder}; run the init action first.")
    out = dict(cfg)
    if has_maps:
        if cfg != NO_CLEANING:
            raise SystemExit(
                f"REFUSING: {folder} already holds maps but no {RECORD_NAME}, so they were "
                "made without cleaning. Cleaned and uncleaned maps would be mixed; use a new "
                "--dataset-dir for the cleaned dataset.")
        out["note"] = "recorded after the fact: maps existed before this record"
    out["sionna_version"] = _sionna_version()
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, RECORD_NAME), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def copy_record(src_folder, dst_folder):
    """Carry the source maps' record into a derived folder (the aggregate stages)."""
    rec = read_record(src_folder)
    if rec is None:
        rec = dict(NO_CLEANING, note="source folder has no record (maps made before it existed)")
    os.makedirs(dst_folder, exist_ok=True)
    with open(os.path.join(dst_folder, RECORD_NAME), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec
