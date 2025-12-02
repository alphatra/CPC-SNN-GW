import os
import argparse
from pathlib import Path
import numpy as np
import h5py

from src.data_handling.gw_data import (
    load_events_yaml,
    setup_gwosc_cache,
    ifo_has_science_data,
    add_sample,
    index_existing_samples,
)
from src.utils.paths import ensure_dir, project_path


# ------------------------------------------------------
#  MAIN BUILDER
# ------------------------------------------------------
def build_multi_event_hdf5(
    events_yaml_path: str,
    output_path: str,
    duration: float,
    positives_per_event: int,
    negatives_per_event: int,
    f_low: float,
    f_high: float,
    bg_range: float,
    margin: float,
    overwrite: bool,
):
    """Main function for building multi-event HDF5 dataset."""
    setup_gwosc_cache()

    # resolve paths
    events_yaml = (
        project_path(events_yaml_path)
        if not os.path.isabs(events_yaml_path)
        else Path(events_yaml_path).resolve()
    )
    out_path = (
        project_path(output_path)
        if not os.path.isabs(output_path)
        else Path(output_path).resolve()
    )

    ensure_dir(out_path.parent)

    events = load_events_yaml(str(events_yaml))
    rng = np.random.default_rng(1234)

    mode = "w" if overwrite or not out_path.exists() else "r+"
    print(f"[INFO] HDF5 mode={mode}, path={out_path}")

    with h5py.File(str(out_path), mode) as h5:

        # handle existing samples
        if mode == "r+":
            existing = index_existing_samples(h5)
            existing_gids = [
                int(k) for k in h5.keys()
                if k.isdigit()
            ]
            gid_counter = max(existing_gids) + 1 if existing_gids else 1

        else:
            existing = {}
            gid_counter = 1

        # ---------------------------------------------------
        # PROCESS EVENTS
        # ---------------------------------------------------
        for ev in events:
            name = ev["name"]
            t_event = float(ev["gps"])
            declared_ifos = ev.get("detectors", ["H1", "L1"])
            src_class = ev.get("class", "BBH")

            print(f"\n[EVENT] {name} @ {t_event}")
            print(f"  declared IFOs: {declared_ifos}  class={src_class}")

            # Check actual availability
            avail_ifos = [
                ifo for ifo in declared_ifos
                if ifo_has_science_data(ifo, t_event, duration)
            ]

            print(f"  available IFOs: {avail_ifos}")

            if not avail_ifos:
                print("  [SKIP] No datasets in GWOSC for declared detectors.")
                continue

            # check what we already have
            stats = existing.get(name, {"pos": 0, "neg": 0})
            need_pos = max(0, positives_per_event - stats["pos"])
            need_neg = max(0, negatives_per_event - stats["neg"])

            if need_pos == 0 and need_neg == 0:
                print(
                    f"  [OK] Already has required samples "
                    f"(pos={stats['pos']}, neg={stats['neg']})."
                )
                continue

            print(
                f"  [PLAN] add +{need_pos} positives, -{need_neg} negatives "
                f"(currently pos={stats['pos']}, neg={stats['neg']})"
            )

            # ----------------------------------------------
            # ADD POSITIVES
            # ----------------------------------------------
            for _ in range(need_pos):
                offset = rng.uniform(-0.3 * duration, +0.3 * duration)
                t_start = t_event - duration / 2 + offset

                gid = f"{gid_counter:06d}"
                ok = add_sample(
                    h5=h5,
                    gid=gid,
                    ifos=avail_ifos,
                    t_start=t_start,
                    duration=duration,
                    is_signal=True,
                    source_class=src_class,
                    event_name=name,
                    f_low=f_low,
                    f_high=f_high,
                )

                if ok:
                    gid_counter += 1

            # ----------------------------------------------
            # ADD NEGATIVES
            # ----------------------------------------------
            for _ in range(need_neg):
                for _tries in range(128):
                    center = rng.uniform(
                        t_event - bg_range,
                        t_event + bg_range
                    )
                    if abs(center - t_event) > margin:
                        t_start = center - duration / 2
                        break
                else:
                    print(
                        f"[WARN] {name}: could not find BG window."
                    )
                    continue

                gid = f"{gid_counter:06d}"

                ok = add_sample(
                    h5=h5,
                    gid=gid,
                    ifos=avail_ifos,
                    t_start=t_start,
                    duration=duration,
                    is_signal=False,
                    source_class="BACKGROUND",
                    event_name=name,
                    f_low=f_low,
                    f_high=f_high,
                )

                if ok:
                    gid_counter += 1

            print(f"  [DONE] added pos={need_pos}, neg={need_neg}")

    print(f"\n[OK] Multi-event dataset saved to {out_path}")


# ------------------------------------------------------
#  CLI
# ------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Build multi-event HDF5 dataset."
    )
    p.add_argument("--events", type=str, default="configs/events.yaml")
    p.add_argument("--output", type=str,
                  default="data/gw_multi_events_sft.h5")

    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--pos-per-event", type=int, default=30)
    p.add_argument("--neg-per-event", type=int, default=400)

    p.add_argument("--f-low", type=float, default=20.0)
    p.add_argument("--f-high", type=float, default=512.0)

    p.add_argument("--bg-range", type=float, default=256.0)
    p.add_argument("--margin", type=float, default=2.0)

    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_multi_event_hdf5(
        events_yaml_path=args.events,
        output_path=args.output,
        duration=args.duration,
        positives_per_event=args.pos_per_event,
        negatives_per_event=args.neg_per_event,
        f_low=args.f_low,
        f_high=args.f_high,
        bg_range=args.bg_range,
        margin=args.margin,
        overwrite=args.overwrite,
    )
