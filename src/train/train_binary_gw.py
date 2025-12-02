import os
import argparse
from math import ceil
from typing import Dict, List, Tuple

import h5py
import yaml
import numpy as np

from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from scipy.signal import stft

from src.utils.paths import project_path, ensure_dir


MULTICLASS_MAP = {
    "BACKGROUND": 0,
    "BBH": 1,
    "BNS": 2,
    "NSBH": 3,
    "GLITCH": 4,
}


def load_events_yaml(path: str) -> List[Dict]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    events = []
    for e in data.get("events", []):
        if not e.get("include", True):
            continue
        if "name" not in e or "gps" not in e:
            continue
        events.append(e)

    if not events:
        raise RuntimeError(f"No usable events found in {path}.")

    return events


def setup_gwosc_cache() -> str:
    cache_dir = project_path("gwosc_cache")
    ensure_dir(cache_dir)
    os.environ["GWOSC_CACHE"] = str(cache_dir)
    return str(cache_dir)


def ifo_has_data(ifo: str, gps: float, duration: float = 4.0) -> bool:
    start = int(gps - duration / 2.0)
    end = int(ceil(gps + duration / 2.0))
    try:
        urls = get_urls(ifo, start, end)
        return bool(urls)
    except Exception:
        return False


def fetch_strain(
    ifo: str,
    t_start: float,
    t_end: float,
    f_low: float,
    f_high: float,
) -> TimeSeries:
    try:
        strain = TimeSeries.fetch_open_data(
            ifo,
            t_start,
            t_end,
            cache=True,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"fetch_open_data failed for {ifo} [{t_start}, {t_end}): {e}"
        )

    target_fs = 4096.0
    if abs(strain.sample_rate.value - target_fs) > 1e-6:
        strain = strain.resample(target_fs)

    strain = strain.bandpass(f_low, f_high)
    return strain


def compute_stft(
    x: np.ndarray,
    fs: float,
    window_sec: float = 0.25,
    overlap_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = int(window_sec * fs)
    noverlap = int(nperseg * overlap_frac)

    f, t, Zxx = stft(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        padded=False,
        boundary=None,
    )

    return f, t, Zxx.T  # [F,T] -> [T,F]


def tf_to_mag_phase_channels(Zxx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mag = np.abs(Zxx)
    pha = np.angle(Zxx)
    cos = np.cos(pha)
    sin = np.sin(pha)
    return (
        mag.astype(np.float32),
        cos.astype(np.float32),
        sin.astype(np.float32),
    )


def build_sample_for_window(
    ifo: str,
    t_start: float,
    duration: float,
    f_low: float,
    f_high: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_end = t_start + duration
    strain = fetch_strain(ifo, t_start, t_end, f_low=f_low, f_high=f_high)
    fs = float(strain.sample_rate.value)
    x = strain.value

    f, _, Zxx = compute_stft(x, fs=fs)
    mag, cos, sin = tf_to_mag_phase_channels(Zxx)

    mask_t = np.ones(mag.shape[0], dtype=np.float32)

    freq_mask = (f >= f_low) & (f <= f_high)
    f_sel = f[freq_mask].astype(np.float32)
    mag = mag[:, freq_mask]
    cos = cos[:, freq_mask]
    sin = sin[:, freq_mask]

    return mag, cos, sin, mask_t, f_sel


def get_multiclass_label(source_class: str, is_signal: bool) -> int:
    if not is_signal:
        return MULTICLASS_MAP["BACKGROUND"]

    key = source_class.upper()
    if key in MULTICLASS_MAP:
        return MULTICLASS_MAP[key]

    return MULTICLASS_MAP["BACKGROUND"]


def add_sample(
    h5: h5py.File,
    gid: str,
    ifos: List[str],
    t_start: float,
    duration: float,
    is_signal: bool,
    source_class: str,
    event_name: str,
    f_low: float,
    f_high: float,
) -> bool:
    per_ifo = {}
    f_common = None

    for ifo in ifos:
        try:
            mag, cos, sin, mask_t, f = build_sample_for_window(
                ifo,
                t_start,
                duration,
                f_low,
                f_high,
            )
        except Exception as e:
            print(f"[WARN] Skipping {gid}/{ifo}: cannot build sample: {e}")
            continue

        if f_common is None:
            f_common = f
        else:
            if f.shape != f_common.shape or not np.allclose(f, f_common):
                print(
                    f"[WARN] Frequency mismatch for {gid}/{ifo}, "
                    f"expected {f_common.shape}, got {f.shape}; skipping this IFO."
                )
                continue

        per_ifo[ifo] = (mag, cos, sin, mask_t)

    if not per_ifo:
        print(f"[WARN] Skipping gid={gid}: no valid IFO data.")
        return False

    for ifo, (mag, cos, sin, mask_t) in per_ifo.items():
        g_ifo = h5.create_group(f"{gid}/{ifo}")
        g_ifo.create_dataset("mag", data=mag, compression="gzip")
        g_ifo.create_dataset("cos", data=cos, compression="gzip")
        g_ifo.create_dataset("sin", data=sin, compression="gzip")
        h5.create_dataset(f"{gid}/mask_{ifo}", data=mask_t, compression="gzip")

    h5.create_dataset(f"{gid}/f", data=f_common, compression="gzip")

    bin_label = np.float32(1.0 if is_signal else 0.0)
    h5.create_dataset(f"{gid}/label", data=bin_label)

    mc_label = np.int32(get_multiclass_label(source_class, is_signal))
    h5.create_dataset(f"{gid}/label_multiclass", data=mc_label)

    g_meta = h5.create_group(f"{gid}/meta")
    g_meta.attrs["event_name"] = str(event_name)
    g_meta.attrs["source_class"] = str(source_class.upper() if is_signal else "BACKGROUND")
    g_meta.attrs["is_signal"] = int(is_signal)
    g_meta.attrs["ifos"] = ",".join(sorted(per_ifo.keys()))

    return True


def index_existing_samples(h5: h5py.File) -> Dict[str, Dict[str, int]]:
    by_event: Dict[str, Dict[str, int]] = {}
    for gid in h5.keys():
        if not gid.isdigit():
            continue
        if "meta" not in h5[gid]:
            continue
        meta = h5[f"{gid}/meta"].attrs
        name = meta.get("event_name", "UNKNOWN")
        is_signal = bool(meta.get("is_signal", 0))
        entry = by_event.setdefault(name, {"pos": 0, "neg": 0})
        if is_signal:
            entry["pos"] += 1
        else:
            entry["neg"] += 1
    return by_event


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
    setup_gwosc_cache()

    from pathlib import Path

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

    if out_path.exists() and not overwrite:
        mode = "r+"
        print(f"[INFO] Updating existing HDF5 at {out_path}")
    else:
        mode = "w"
        print(f"[INFO] Creating new HDF5 at {out_path}")

    with h5py.File(str(out_path), mode) as h5:
        if mode == "r+":
            existing = index_existing_samples(h5)
            existing_gids = [int(k) for k in h5.keys() if k.isdigit()]
            gid_counter = max(existing_gids) + 1 if existing_gids else 1
        else:
            existing = {}
            gid_counter = 1

        for ev in events:
            name = ev["name"]
            t_event = float(ev["gps"])
            declared_ifos = ev.get("detectors", ["H1", "L1"])

            avail_ifos = [
                ifo for ifo in declared_ifos if ifo_has_data(ifo, t_event, duration=duration)
            ]

            if not avail_ifos:
                print(
                    f"[WARN] Skipping {name}: no GWOSC data near gps={t_event} "
                    f"for declared detectors {declared_ifos}."
                )
                continue

            stats = existing.get(name, {"pos": 0, "neg": 0})
            need_pos = max(0, positives_per_event - stats["pos"])
            need_neg = max(0, negatives_per_event - stats["neg"])

            if need_pos == 0 and need_neg == 0:
                print(
                    f"[INFO] {name}: already has required samples "
                    f"(pos={stats['pos']}, neg={stats['neg']})."
                )
                continue

            src_class = ev.get("class", "BBH")

            print(
                f"[INFO] {name} ({src_class}) @ {t_event} | IFO={avail_ifos} | "
                f"adding +{need_pos}, -{need_neg}"
            )

            for _ in range(need_pos):
                offset = rng.uniform(-0.3 * duration, 0.3 * duration)
                t_start = t_event - duration / 2.0 + offset
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

            for _ in range(need_neg):
                for _tries in range(128):
                    center = rng.uniform(t_event - bg_range, t_event + bg_range)
                    if abs(center - t_event) > margin:
                        t_start = center - duration / 2.0
                        break
                else:
                    print(f"[WARN] {name}: could not find BG window after many tries.")
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

    print(f"[INFO] Multi-event HDF5 dataset saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-event HDF5 dataset from GWOSC real data (incremental)."
    )
    parser.add_argument(
        "--events",
        type=str,
        default="configs/events.yaml",
        help="Ścieżka do events.yaml (relatywna lub absolutna)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gw_multi_events_sft.h5",
        help="Ścieżka do wyjściowego HDF5",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Długość okna [s]",
    )
    parser.add_argument(
        "--pos-per-event",
        type=int,
        default=10,
        help="Liczba dodatnich próbek na event",
    )
    parser.add_argument(
        "--neg-per-event",
        type=int,
        default=200,
        help="Liczba próbek tła na event",
    )
    parser.add_argument(
        "--f-low",
        type=float,
        default=20.0,
        help="Dolne ograniczenie pasma [Hz]",
    )
    parser.add_argument(
        "--f-high",
        type=float,
        default=512.0,
        help="Górne ograniczenie pasma [Hz]",
    )
    parser.add_argument(
        "--bg-range",
        type=float,
        default=256.0,
        help="Zasięg losowania tła wokół eventu [s]",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help="Martwa strefa wokół eventu [s]",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Nadpisz istniejący plik HDF5 zamiast aktualizować",
    )
    return parser.parse_args()


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
        overwrite=bool(args.overwrite),
    )