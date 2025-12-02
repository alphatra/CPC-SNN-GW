import os
import numpy as np
from math import ceil
from typing import Dict, List, Tuple

import h5py
import yaml
from scipy.signal import stft
from gwpy.timeseries import TimeSeries
from gwosc.datasets import find_datasets
from gwosc.locate import get_urls

from src.utils.paths import project_path, ensure_dir, CACHE_DIR


# -----------------------------------------------------
#  CONSTANTS
# -----------------------------------------------------
MULTICLASS_MAP = {
    "BACKGROUND": 0,
    "BBH": 1,
    "BNS": 2,
    "NSBH": 3,
    "GLITCH": 4,
}


# -----------------------------------------------------
#  YAML EVENTS LOADING
# -----------------------------------------------------
def load_events_yaml(path: str) -> List[Dict]:
    """Load events from YAML file, ignoring disabled entries."""
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


# -----------------------------------------------------
#  CACHE SETUP
# -----------------------------------------------------
def setup_gwosc_cache() -> str:
    ensure_dir(CACHE_DIR)
    os.environ["GWOSC_CACHE"] = str(CACHE_DIR)
    return str(CACHE_DIR)


# -----------------------------------------------------
#  SEGMENT & IFO CHECKING
# -----------------------------------------------------
def ifo_has_science_data(ifo: str, gps: float, duration: float = 4.0) -> bool:
    """
    Check if an interferometer has data around the event.
    Works by checking if datasets exist for a segment.
    """
    start = int(gps - duration / 2)
    end = int(ceil(gps + duration / 2))

    try:
        urls = get_urls(ifo, start, end)
        return bool(urls)
    except Exception:
        return False


# -----------------------------------------------------
#  FETCH STRAIN
# -----------------------------------------------------
def fetch_strain(
    ifo: str,
    t_start: float,
    t_end: float,
    f_low: float,
    f_high: float,
) -> TimeSeries:
    """Download, resample, and band-pass filter GW strain data."""
    try:
        strain = TimeSeries.fetch_open_data(
            ifo, t_start, t_end, cache=True, verbose=False
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


# -----------------------------------------------------
#  STFT & TF CHANNELS
# -----------------------------------------------------
def compute_stft(
    x: np.ndarray,
    fs: float,
    window_sec: float = 0.25,
    overlap_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT for TF representation."""
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

    return f, t, Zxx.T  # [T, F]


def tf_to_mag_phase_channels(Zxx: np.ndarray):
    """Convert complex STFT into magnitude/cos/sin channels."""
    mag = np.abs(Zxx)
    pha = np.angle(Zxx)
    cos = np.cos(pha)
    sin = np.sin(pha)

    return (
        mag.astype(np.float32),
        cos.astype(np.float32),
        sin.astype(np.float32),
    )


# -----------------------------------------------------
#  SAMPLE FOR A SINGLE WINDOW
# -----------------------------------------------------
def build_sample_for_window(
    ifo: str,
    t_start: float,
    duration: float,
    f_low: float,
    f_high: float,
):
    """Build TF representation for given IFO and time window."""
    t_end = t_start + duration
    strain = fetch_strain(ifo, t_start, t_end, f_low, f_high)
    fs = float(strain.sample_rate.value)
    x = strain.value

    f, _, Zxx = compute_stft(x, fs)
    mag, cos, sin = tf_to_mag_phase_channels(Zxx)

    # frequency trimming
    freq_mask = (f >= f_low) & (f <= f_high)
    f_sel = f[freq_mask].astype(np.float32)

    mag = mag[:, freq_mask]
    cos = cos[:, freq_mask]
    sin = sin[:, freq_mask]

    mask_t = np.ones(mag.shape[0], dtype=np.float32)

    return mag, cos, sin, mask_t, f_sel


# -----------------------------------------------------
#  LABEL UTILS
# -----------------------------------------------------
def get_multiclass_label(source_class: str, is_signal: bool) -> int:
    if not is_signal:
        return MULTICLASS_MAP["BACKGROUND"]
    key = source_class.upper()
    return MULTICLASS_MAP.get(key, MULTICLASS_MAP["BACKGROUND"])


# -----------------------------------------------------
#  HDF5 INSERTION
# -----------------------------------------------------
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
    """Add multi-IFO sample into HDF5."""
    per_ifo = {}
    f_common = None

    for ifo in ifos:
        try:
            mag, cos, sin, mask_t, f = build_sample_for_window(
                ifo, t_start, duration, f_low, f_high
            )
        except Exception as e:
            print(f"[WARN] Skipping {gid}/{ifo}: cannot build sample: {e}")
            continue

        if f_common is None:
            f_common = f
        else:
            if f.shape != f_common.shape or not np.allclose(f, f_common):
                print(
                    f"[WARN] Frequency mismatch for {gid}/{ifo}; skipping."
                )
                continue

        per_ifo[ifo] = (mag, cos, sin, mask_t)

    if not per_ifo:
        print(f"[WARN] Skipping gid={gid}: no valid IFO data.")
        return False

    # write IFO groups
    for ifo, (mag, cos, sin, mask_t) in per_ifo.items():
        g = h5.create_group(f"{gid}/{ifo}")
        g.create_dataset("mag", data=mag, compression="gzip")
        g.create_dataset("cos", data=cos, compression="gzip")
        g.create_dataset("sin", data=sin, compression="gzip")
        h5.create_dataset(f"{gid}/mask_{ifo}", data=mask_t, compression="gzip")

    h5.create_dataset(f"{gid}/f", data=f_common, compression="gzip")

    # labels
    h5.create_dataset(f"{gid}/label", data=np.float32(is_signal))
    mc = get_multiclass_label(source_class, is_signal)
    h5.create_dataset(f"{gid}/label_multiclass", data=np.int32(mc))

    # metadata
    meta = h5.create_group(f"{gid}/meta")
    meta.attrs["event_name"] = str(event_name)
    meta.attrs["source_class"] = str(source_class.upper() if is_signal else "BACKGROUND")
    meta.attrs["is_signal"] = int(is_signal)
    meta.attrs["ifos"] = ",".join(sorted(per_ifo.keys()))

    return True


# -----------------------------------------------------
#  INDEX EXISTING SAMPLES
# -----------------------------------------------------
def index_existing_samples(h5: h5py.File) -> Dict[str, Dict[str, int]]:
    by_event = {}
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

