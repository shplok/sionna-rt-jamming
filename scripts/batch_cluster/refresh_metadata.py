#!/usr/bin/env python3
"""Refresh the density-dependent metadata without touching the RSS cubes.

Sensor density is preprocessing, not physics: `rss.npy` and `rss_aggregated.npy` are
built from `position_ids` / `jammers` and `noise_seed` alone (see run_static_aggregation
and run_aggregation_from_splits in main_batch_cluster.py). Density, num_sensors and
sensor_seed are only ever *recorded* alongside them.

The aggregate stages write that metadata inside the same loop that opens the RSS memmap
with mode="w+", so re-running them to pick up a new density ladder would rewrite ~87 GB
to change ~10 MB. This script rewrites only the metadata:

    multi_static_jammers/{split}/meta.npz
    multi_trajectory_jammers/{split}/{scenario_id}/scenario_summary.json

Before writing anything it asserts that every field the frozen RSS actually depends on is
unchanged in splits.json. If that check fails the RSS on disk is stale for these splits
and the script refuses -- that is a real regeneration, not a metadata refresh.
"""
import argparse
import json
import os

import numpy as np


def _fail(msg):
    raise SystemExit(
        f"REFUSING TO WRITE: {msg}\n"
        "The RSS cubes on disk were generated from different sample specs, so this is not a\n"
        "metadata-only change. Re-run the aggregate stages instead (and expect ~87 GB of I/O)."
    )


def refresh_static(dataset_dir, splits, check_only):
    root = os.path.join(dataset_dir, "multi_static_jammers")
    touched = []
    for split, samples in splits["static"]["samples"].items():
        meta_path = os.path.join(root, split, "meta.npz")
        if not os.path.exists(meta_path):
            print(f"  {split}: no meta.npz, skipping")
            continue
        old = np.load(meta_path)

        ks = np.asarray([len(s["position_ids"]) for s in samples], np.int8)
        flat, off = [], [0]
        for s in samples:
            flat.extend(s["position_ids"])
            off.append(len(flat))
        flat = np.asarray(flat, np.int32)
        off = np.asarray(off, np.int64)
        nseed = np.asarray([s["noise_seed"] for s in samples], np.int64)

        if len(ks) != len(old["num_jammers"]):
            _fail(f"{split}: sample count {len(ks)} != {len(old['num_jammers'])} on disk")
        for name, new_v, old_v in (("num_jammers", ks, old["num_jammers"]),
                                   ("position_ids", flat, old["position_ids"]),
                                   ("position_offsets", off, old["position_offsets"]),
                                   ("noise_seed", nseed, old["noise_seed"])):
            if not np.array_equal(new_v, old_v):
                _fail(f"{split}: {name} differs from the on-disk metadata")

        if check_only:
            print(f"  {split}: OK, RSS still matches ({len(ks)} samples)")
            continue
        np.savez_compressed(
            meta_path,
            num_jammers=ks,
            position_ids=flat,
            position_offsets=off,
            sensor_density_pct=np.asarray([s["sensor_density_pct"] for s in samples], np.float32),
            num_sensors=np.asarray([s["num_sensors"] for s in samples], np.int32),
            sensor_seed=np.asarray([s["sensor_seed"] for s in samples], np.int64),
            noise_seed=nseed,
        )
        touched.append(meta_path)
        print(f"  {split}: meta.npz rewritten ({len(ks)} samples)")
    return touched


def refresh_tracking(dataset_dir, splits, check_only):
    root = os.path.join(dataset_dir, "multi_trajectory_jammers")
    touched = []
    for split, scens in splits["scenarios"].items():
        n_ok = 0
        for sc in scens:
            path = os.path.join(root, split, sc["scenario_id"], "scenario_summary.json")
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if doc.get("num_jammers") != len(sc["jammers"]):
                _fail(f"{sc['scenario_id']}: num_jammers differs from the on-disk summary")
            if doc.get("noise_seed") != sc["noise_seed"]:
                _fail(f"{sc['scenario_id']}: noise_seed differs from the on-disk summary")
            if check_only:
                n_ok += 1
                continue
            doc["sensor_density_pct"] = sc["sensor_density_pct"]
            doc["num_sensors"] = sc["num_sensors"]
            doc["sensor_seed"] = sc["sensor_seed"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            touched.append(path)
            n_ok += 1
        print(f"  {split}: {n_ok} scenario_summary.json {'checked' if check_only else 'rewritten'}")
    return touched


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--splits", default=None, help="Default: <dataset-dir>/splits.json")
    p.add_argument("--check-only", action="store_true",
                   help="Verify the RSS still matches splits.json; write nothing.")
    args = p.parse_args()
    splits_path = args.splits or os.path.join(args.dataset_dir, "splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    print(f"Densities in {splits_path}: {splits.get('sensor_densities_pct')}")
    print("Detector samples:")
    a = refresh_static(args.dataset_dir, splits, args.check_only)
    print("Tracking scenarios:")
    b = refresh_tracking(args.dataset_dir, splits, args.check_only)
    print(f"\n{'Checked' if args.check_only else 'Rewrote'} {len(a) + len(b)} files. "
          "RSS cubes untouched.")


if __name__ == "__main__":
    main()
