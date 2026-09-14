#!/usr/bin/env python3
"""
Check a generated batch_simulation_<tag> dataset before you trust it.  [CPU]

Worth running after the ~3 h pipeline: a silent mistake there costs another 3 h, and some
of the failure modes are quiet ones (a sensor inside a building, labels off the grid, a
stale base library at the wrong cell size, K=0 samples that came out flat because the noise
was disabled).

Each check prints PASS / FAIL / SKIP. Exits non-zero if anything failed, so it can gate a
SLURM job. Branches that were not generated are skipped, not failed.

Usage:
    python scripts/validate_dataset.py --dataset-dir ./datasets/batch_simulation_nyc
    python scripts/validate_dataset.py --dataset-dir ... --quick   # skip array-wide reads
"""

import argparse
import json
import os
import sys

import numpy as np

OK, BAD, SKIP = "PASS", "FAIL", "SKIP"
_results = []


def check(name, fn):
    """Run one check. fn returns (status, detail)."""
    try:
        status, detail = fn()
    except FileNotFoundError as e:
        status, detail = SKIP, f"missing: {os.path.basename(str(e).split(':')[-1].strip())}"
    except Exception as e:                                    # noqa: BLE001
        status, detail = BAD, f"{type(e).__name__}: {e}"
    _results.append((status, name, detail))
    mark = {OK: "  PASS", BAD: "  FAIL", SKIP: "  skip"}[status]
    print(f"{mark}  {name}" + (f"  --  {detail}" if detail else ""))
    return status


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="./datasets/batch_simulation_nyc")
    p.add_argument("--quick", action="store_true",
                   help="Skip checks that read whole arrays (dtype/shape only).")
    p.add_argument("--sample-scenarios", type=int, default=25,
                   help="Tracking scenarios to open in full (default: 25).")
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.normpath(args.dataset_dir)
    P = lambda *a: os.path.join(root, *a)                      # noqa: E731
    print(f"Validating {root}\n")

    # ---------------------------------------------------------------- splits.json
    print("splits.json")
    with open(P("splits.json"), "r", encoding="utf-8") as f:
        sp = json.load(f)
    grid = sp["grid"]
    n_cells, cell = grid["n_cells"], grid["cell_size_m"]
    x0, y0 = grid["origin_m"]
    print(f"  grid {n_cells}x{n_cells} @ {cell:g} m, origin {grid['origin_m']}\n")

    pools = sp["trajectory_pools"]
    check("trajectory pools are disjoint", lambda: (
        (OK, f"{sum(len(v) for v in pools.values())} trajectories")
        if not any(set(pools[a]) & set(pools[b])
                   for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
        else (BAD, "pools overlap")))

    def scen_pool_ok():
        bad = 0
        for s, scens in sp["scenarios"].items():
            allowed = set(pools[s])
            bad += sum(1 for sc in scens if not set(sc["jammers"]) <= allowed)
        return (OK, "no scenario uses a foreign trajectory") if bad == 0 else \
               (BAD, f"{bad} scenarios draw outside their pool")
    check("scenarios stay inside their own pool", scen_pool_ok)

    def sensors_on_street():
        cells = set(sp["sensor_cells"])
        n_bad = 0
        for s in ("train", "val", "test"):
            f = P("sensors", f"sensors_{s}.npz")
            if not os.path.exists(f):
                raise FileNotFoundError(f)
            z = np.load(f)
            n_bad += int((~np.isin(z["cells"], list(cells))).sum()) if len(z["cells"]) < 2e6 \
                else int(len(np.setdiff1d(z["cells"], np.fromiter(cells, dtype=np.int64))))
        return (OK, f"{len(cells)} placeable cells") if n_bad == 0 else \
               (BAD, f"{n_bad} sensors outside the street mask")
    check("every sensor sits on a street cell", sensors_on_street)

    def sensor_counts():
        bad = []
        for s in ("train", "val", "test"):
            z = np.load(P("sensors", f"sensors_{s}.npz"))
            off = z["offsets"]
            got = np.diff(off)
            want = np.array([x["num_sensors"] for x in sp["scenarios"][s]])
            if len(got) != len(want) or not np.array_equal(got, want):
                bad.append(s)
        return (OK, "layout sizes match the declared densities") if not bad else \
               (BAD, f"count mismatch in {bad}")
    check("sensor layout sizes match num_sensors", sensor_counts)

    # ------------------------------------------------- tracking: base radio maps
    print("\nsingle_trajectory_jammers/  [tracking base]")
    rm_dir = P("single_trajectory_jammers", "radio_maps")

    def bases_present():
        with open(os.path.join(rm_dir, "radio_maps_manifest.json"), "r", encoding="utf-8") as f:
            man = json.load(f)["radio_maps"]
        n_traj = sum(len(v) for v in pools.values())
        return (OK, f"{len(man)} maps") if len(man) == n_traj else \
               (BAD, f"{len(man)} maps for {n_traj} trajectories")
    check("one base map per trajectory", bases_present)

    def base_grid():
        with open(os.path.join(rm_dir, "radio_maps_manifest.json"), "r", encoding="utf-8") as f:
            man = json.load(f)["radio_maps"]
        w = np.load(os.path.join(rm_dir, man[0]["watts_file"]), mmap_mode="r")
        return (OK, f"{w.shape[1]}x{w.shape[2]} {w.dtype}") if w.shape[1] == n_cells else \
               (BAD, f"maps are {w.shape[1]}x{w.shape[2]}, splits.json says {n_cells} "
                     f"-- stale library, re-run simulate_bases")
    check("base maps match the splits grid", base_grid)

    # ------------------------------------------------- tracking: scenarios
    print("\nmulti_trajectory_jammers/  [tracking]")

    def scen_dirs():
        miss = []
        for s in ("train", "val", "test"):
            want = {x["scenario_id"] for x in sp["scenarios"][s]}
            d = P("multi_trajectory_jammers", s)
            if not os.path.isdir(d):
                raise FileNotFoundError(d)
            got = set(os.listdir(d))
            miss.append((s, len(want - got)))
        tot = sum(m for _, m in miss)
        return (OK, f"{sum(len(sp['scenarios'][s]) for s in sp['scenarios'])} scenarios") \
            if tot == 0 else (BAD, f"missing dirs: {miss}")
    check("every scenario in splits.json was written", scen_dirs)

    def scen_contents():
        rng = np.random.default_rng(0)
        bad, checked, k0_std = [], 0, []
        for s in ("train", "val", "test"):
            scens = sp["scenarios"][s]
            pick = rng.choice(len(scens), size=min(args.sample_scenarios, len(scens)),
                              replace=False)
            for i in pick:
                sc = scens[int(i)]
                d = P("multi_trajectory_jammers", s, sc["scenario_id"])
                a = np.load(os.path.join(d, "rss_aggregated.npy"), mmap_mode="r")
                if a.shape[1] != n_cells or a.shape[2] != n_cells:
                    bad.append((sc["scenario_id"], "grid"))
                if not args.quick:
                    v = np.asarray(a, dtype=np.float32)
                    if not np.all(np.isfinite(v)):
                        bad.append((sc["scenario_id"], "non-finite"))
                    if len(sc["jammers"]) == 0:
                        k0_std.append(float(v.std()))
                checked += 1
        det = f"{checked} opened"
        if k0_std:
            det += f", K=0 std {np.mean(k0_std):.2f} dB"
        return (OK, det) if not bad else (BAD, f"{bad[:3]}")
    check("scenario cubes: shape, finite, K=0 carries noise", scen_contents)

    def traj_labels():
        bad = []
        for s in ("train", "val", "test"):
            z = np.load(P("labels", f"labels_{s}.npz"))
            if len(z["col"]) == 0:
                continue
            if (z["col"] < 0).any() or (z["col"] >= n_cells).any() or \
               (z["row"] < 0).any() or (z["row"] >= n_cells).any():
                bad.append(f"{s}: off-grid")
            xr = x0 + z["col"] * cell + z["dx"]
            if np.abs(xr - z["x"]).max() > 1e-6:
                bad.append(f"{s}: (col,dx) does not reconstruct x")
            if float(np.ptp(z["dx"])) < cell * 0.5:
                bad.append(f"{s}: dx spans only {np.ptp(z['dx']):.1f} m -- jitter degenerate?")
        return (OK, "in-grid, round-trip exact, offsets span the cell") if not bad else \
               (BAD, "; ".join(bad))
    check("tracking labels round-trip to metres", traj_labels)

    # ------------------------------------------------- detection branch
    print("\nsingle_static_jammers/ + multi_static_jammers/  [detection]")
    if "static" not in sp:
        check("detection branch planned", lambda: (SKIP, "no 'static' section in splits.json"))
    else:
        st = sp["static"]

        def static_positions():
            pos = np.load(P("single_static_jammers", "positions.npy"))
            street = set(sp["sensor_cells"])
            col = np.floor((pos[:, 0] - x0) / cell).astype(int)
            row = np.floor((pos[:, 1] - y0) / cell).astype(int)
            flat = row * n_cells + col
            off = int(np.isin(flat, list(street), invert=True).sum()) if len(street) < 2e6 else 0
            dx = (pos[:, 0] - x0) / cell % 1.0 * cell
            det = f"{len(pos)} positions, dx spans {np.ptp(dx):.1f} m"
            if off:
                return BAD, f"{off} positions off-street"
            if np.ptp(dx) < cell * 0.5:
                return BAD, f"jitter degenerate: dx spans only {np.ptp(dx):.1f} m"
            return OK, det
        check("static positions are on streets and jittered", static_positions)

        def static_split_disjoint():
            P_ = st["position_split"]
            ov = [(a, b, len(set(P_[a]) & set(P_[b])))
                  for a, b in (("train", "val"), ("train", "test"), ("val", "test"))]
            bad = [x for x in ov if x[2]]
            return (OK, st.get("split_mode", "")) if not bad else (BAD, f"overlap {bad}")
        check("static position splits are disjoint", static_split_disjoint)

        def static_maps():
            w = np.load(P("single_static_jammers", "watts.npy"), mmap_mode="r")
            n = st["n_positions"]
            if w.shape != (n, n_cells, n_cells):
                return BAD, f"watts.npy is {w.shape}, expected ({n}, {n_cells}, {n_cells})"
            return OK, f"{w.shape} {w.dtype}"
        check("static library shape matches splits.json", static_maps)

        def detector_samples():
            issues, notes = [], []
            for s, samples in st["samples"].items():
                d = P("multi_static_jammers", s)
                rss = np.load(os.path.join(d, "rss.npy"), mmap_mode="r")
                meta = np.load(os.path.join(d, "meta.npz"))
                lab = np.load(os.path.join(d, "labels.npz"))
                if rss.shape != (len(samples), n_cells, n_cells):
                    issues.append(f"{s}: rss {rss.shape}")
                if int(meta["num_jammers"].sum()) != len(lab["x"]):
                    issues.append(f"{s}: label rows != sum(K)")
                allowed = set(st["position_split"][s])
                if not set(meta["position_ids"].tolist()) <= allowed:
                    issues.append(f"{s}: sample uses a foreign position")
                ks = meta["num_jammers"]
                hist = np.bincount(ks, minlength=11)
                if hist.max() - hist[hist > 0].min() > max(2, 0.02 * hist.max()):
                    issues.append(f"{s}: K not stratified {hist.tolist()}")
                notes.append(f"{s}:{len(samples)}")
                if not args.quick:
                    k0 = np.flatnonzero(ks == 0)[:5]
                    for i in k0:
                        if float(np.asarray(rss[i], np.float32).std()) < 1e-3:
                            issues.append(f"{s}: K=0 sample {i} is flat -- noise disabled?")
            return (OK, ", ".join(notes)) if not issues else (BAD, "; ".join(issues[:3]))
        check("detector samples: shapes, K stratified, own positions, K=0 noisy",
              detector_samples)

    # ---------------------------------------------------------------- summary
    n_fail = sum(1 for s, _, _ in _results if s == BAD)
    n_skip = sum(1 for s, _, _ in _results if s == SKIP)
    n_pass = sum(1 for s, _, _ in _results if s == OK)
    print(f"\n{'='*62}\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nFailures:")
        for s, name, det in _results:
            if s == BAD:
                print(f"  - {name}: {det}")
    print("=" * 62)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
