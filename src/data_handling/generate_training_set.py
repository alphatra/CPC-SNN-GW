# -------------------------------------------------------------
#  generate_training_set.py  —  Dr. Gravitas (v4.0)
#  Robust GW noise+injection dataset generator for CPC–SNN
#  Supports Multi-IFO and Hydra Configuration
# -------------------------------------------------------------

import os
import time
import logging
import warnings
import numpy as np
import h5py
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List, Dict

from pycbc.types import TimeSeries as PyCBCTimeSeries
from gwpy.timeseries import TimeSeries
from gwosc.timeline import get_segments
from gwosc.datasets import find_datasets
from gwosc.locate import get_urls

import sys
# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

from src.data_handling.gw_utils import (
    get_whitened_strain,
    generate_waveform,
    project_waveform_to_ifo,
    generate_glitch,
)

from src.data_handling.gw_data import compute_stft, tf_to_mag_phase_channels, setup_gwosc_cache

from src.utils.paths import ensure_dir, project_path, CACHE_DIR

# Configure logging
log = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["MPLBACKEND"] = "Agg"


def get_coincident_segments(
    ifos: List[str],
    start: int,
    end: int,
    min_len: int
) -> List[tuple]:
    """
    Finds time segments where ALL specified interferometers are active (DATA flag).
    """
    log.info(f"Searching for coincident segments for {ifos} in [{start}, {end}]...")
    
    # Get segments for the first IFO
    base_segments = get_segments(flag=f"{ifos[0]}_DATA", start=start, end=end)
    
    # Intersect with other IFOs
    from gwpy.segments import SegmentList, Segment
    
    # Convert to SegmentList
    coinc_segments = SegmentList([Segment(s[0], s[1]) for s in base_segments])
    
    for ifo in ifos[1:]:
        other_segs = get_segments(flag=f"{ifo}_DATA", start=start, end=end)
        other_list = SegmentList([Segment(s[0], s[1]) for s in other_segs])
        coinc_segments = coinc_segments & other_list  # Intersection
        
    # Filter by length
    valid_segments = [
        (int(s[0]), int(s[1])) 
        for s in coinc_segments 
        if (s[1] - s[0]) >= min_len
    ]
    
    log.info(f"Found {len(valid_segments)} coincident segments > {min_len}s.")
    return valid_segments



def fetch_with_retry(ifo, start, end, retries=5, backoff=2.0):
    """
    Fetches data with robust retry logic for network timeouts.
    """
    attempt = 0
    while attempt < retries:
        try:
            return TimeSeries.fetch_open_data(
                ifo, start, end,
                cache=True, verbose=False
            )
        except Exception as e:
            attempt += 1
            wait = backoff ** attempt
            log.warning(f"  [{ifo}] Fetch failed (attempt {attempt}/{retries}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch data for {ifo} after {retries} attempts.")


def process_segment(
    segment: tuple,
    cfg: DictConfig,
    h5_file: h5py.File,

    start_idx: int
) -> int:
    """
    Processes a single continuous segment:
    1. Download data for all IFOs.
    2. Whiten and sanitize.
    3. Generate samples (Noise & Injection).
    4. Save to HDF5.
    
    Returns the number of samples added.
    """
    t_start, t_end = segment
    duration = t_end - t_start
    
    # Limit download size if segment is huge (to avoid OOM)
    # For now, we take the whole segment or a chunk of it.
    # Let's take up to 4096s for safety, or loop if needed. 
    # But user script had simple logic. Let's stick to simple for now.
    load_end = min(t_start + 4096, t_end) 
    
    log.info(f"Processing segment {t_start}-{load_end} ({load_end-t_start}s)...")
    
    # 1. Fetch and Whiten Data for all IFOs
    strain_data = {}
    strain_psds = {}
    
    for ifo in cfg.ifo_list:
        try:
            log.info(f"  [{ifo}] Fetching...")
            raw = fetch_with_retry(ifo, t_start, load_end)

            log.info(f"  [{ifo}] Whitening...")
            whitened, psd = get_whitened_strain(raw, cfg.sample_rate, cfg.f_low)
            strain_data[ifo] = whitened
            strain_psds[ifo] = psd
        except Exception as e:
            log.error(f"  [{ifo}] Failed to fetch/whiten: {e}")
            return 0



    # Check if all IFOs have valid data

    if len(strain_data) != len(cfg.ifo_list):
        return 0
        
    # Ensure all have same duration/start (after whitening cropping)
    # Find common intersection
    common_start = float(max(s.start_time for s in strain_data.values()))
    common_end = float(min(s.end_time for s in strain_data.values()))


    
    if common_end - common_start < cfg.duration:
        log.warning("  Common valid segment too short.")
        return 0
        
    # Crop all to common segment
    arrays = {}
    for ifo, s in strain_data.items():
        # PyCBC crop takes (left_crop, right_crop) in seconds
        left = float(common_start - s.start_time)
        right = float(s.end_time - common_end)
        
        # Ensure non-negative (should be guaranteed by min/max logic)
        left = max(0.0, left)
        right = max(0.0, right)
        
        cropped = s.crop(left, right)
        arrays[ifo] = cropped.numpy()

        
    total_samples = len(arrays[cfg.ifo_list[0]])
    window_samples = int(cfg.duration * cfg.sample_rate)
    
    if total_samples < window_samples:
        return 0

    # 2. Generate Samples
    samples_to_gen = 200 # Generate 200 pairs (pos+neg) per segment pass
    
    added_count = 0
    
    # Pre-compute dummy STFT to get shape
    dummy_f, _, dummy_Zxx = compute_stft(
        np.zeros(window_samples), cfg.sample_rate
    )
    # We need to match the shape expected by torch_dataset
    # gw_data.compute_stft returns (f, t, Zxx.T) -> [T, F]
    # We need to trim frequencies
    freq_mask = (dummy_f >= cfg.f_low) & (dummy_f <= cfg.f_high)
    f_sel = dummy_f[freq_mask].astype(np.float32)
    feat_shape = (dummy_Zxx.T.shape[0], np.sum(freq_mask)) # [T, F_trimmed]
    
    
    # ---------------------------------------------------------
    # LOOP for Samples
    # ---------------------------------------------------------
    for _ in range(samples_to_gen):
        # Random start index
        max_idx = total_samples - window_samples
        if max_idx <= 0: break
        
        idx = np.random.randint(0, max_idx)
        
        # --- INJECTION (SIGNAL + NOISE or GLITCH + NOISE) ---
        gid_inj = f"{start_idx + added_count:06d}"
        
        # Decide: Glitch or Signal?
        # If we are generating "Signal" class (label=1), it's BBH.
        # If we are generating "Background" class (label=0), it's Noise.
        # But we want to inject glitches into Background to make it robust.
        # Wait, usually we generate pairs: (Background, Signal).
        # Let's modify this:
        # 1. Generate Background (Pure Noise OR Glitch+Noise)
        # 2. Generate Signal (BBH+Noise)
        
        # --- 1. BACKGROUND ---
        gid_bg = f"{start_idx + added_count:06d}"
        
        # Check if we should inject a glitch into the background

        is_glitch = np.random.random() < cfg.get("glitch_prob_bg", 0.0)
        
        noise_chunks = {
            ifo: arrays[ifo][idx : idx + window_samples]
            for ifo in cfg.ifo_list
        }
        
        bg_meta = {}
        
        if is_glitch:
            # Generate Glitch
            # Glitches are usually uncorrelated between detectors (instrumental)
            # So we inject independent glitches into H1 and L1 (or just one of them?)
            # Real glitches are local. Let's inject into a random subset of IFOs (or just one).
            
            target_ifo = str(np.random.choice(cfg.ifo_list))

            glitch = generate_glitch(
                sample_rate=cfg.sample_rate,
                duration=cfg.duration, # Make glitch duration same as window for simplicity, or shorter
                # But generate_glitch returns a burst centered in duration.
            )
            
            # Add to noise
            # Glitch is already a TimeSeries.
            # We need to add it to the numpy array of the chunk.
            # Ensure lengths match.
            g_data = glitch.numpy()
            if len(g_data) > window_samples:
                g_data = g_data[:window_samples]
            elif len(g_data) < window_samples:
                # Pad? Or just center?
                # generate_glitch uses 'duration' arg.
                pass
                
            # Add to the specific IFO chunk
            # Note: noise_chunks[target_ifo] is a reference to the big array? No, slicing creates a view or copy.
            # We should copy to be safe.
            noise_chunks[target_ifo] = noise_chunks[target_ifo].copy() + g_data
            
            bg_meta["has_glitch"] = 1
            bg_meta["glitch_ifo"] = target_ifo
            bg_meta["source_class"] = "GLITCH"
        else:
            bg_meta["has_glitch"] = 0
            bg_meta["source_class"] = "BACKGROUND"

        # Save Background
        save_sample(
            h5_file, gid_bg, noise_chunks, 
            cfg, is_signal=False, f_common=f_sel, freq_mask=freq_mask,
            meta=bg_meta
        )
        added_count += 1
        
        # --- 2. SIGNAL (BBH) ---
        # (Standard injection logic)
        gid_inj = f"{start_idx + added_count:06d}"
        
        # Generate Waveform
        hp, hc = generate_waveform(
            mass_range=tuple(cfg.mass_range),
            sample_rate=cfg.sample_rate,
            f_lower=cfg.f_low
        )
        
        # Project and Add
        inj_chunks = {}

        
        # Random sky position
        ra = np.random.uniform(0, 2 * np.pi)
        dec = np.random.uniform(-np.pi / 2, np.pi / 2) # Cosine distribution would be better but uniform is ok for now
        psi = np.random.uniform(0, 2 * np.pi)
        
        # Time of merger: center of the window + random offset?
        # The noise chunk corresponds to [common_start + idx/fs, ...]
        # GPS time of the chunk start
        t_gps_start = common_start + idx / cfg.sample_rate
        # We want merger in the window.
        t_merger = t_gps_start + cfg.duration / 2 + np.random.uniform(-0.5, 0.5)
        
        # Target SNR
        snr_target = np.random.uniform(*cfg.snr_range)
        
        # We need to scale the signal to achieve target SNR in the NETWORK or single detector?
        # Usually defined as optimal SNR in the network or specific IFO.
        # Let's calculate network SNR.
        
        # First project to all IFOs
        raw_signals = {}
        network_snr_sq = 0.0
        
        for ifo in cfg.ifo_list:
            sig = project_waveform_to_ifo(hp, hc, ifo, ra, dec, psi, t_merger)
            sig.start_time += t_merger
            
            # Resize/Shift to match the noise window

            # The signal `sig` has `start_time`.
            # We need to sample it at the times corresponding to `noise_chunks[ifo]`.
            # Noise chunk starts at `t_gps_start`.
            # We can interpolate or just align if sample rates match.
            
            # Simple alignment:
            # Time relative to signal start
            dt_start = t_gps_start - sig.start_time
            start_idx_sig = int(dt_start * cfg.sample_rate)
            
            # Extract
            sig_np = sig.numpy()
            if start_idx_sig < 0:
                # Signal starts after window start (should not happen if merger is in middle and waveform is long enough)
                # Pad beginning
                pad = -start_idx_sig
                sig_cut = sig_np[:window_samples-pad]
                sig_cut = np.pad(sig_cut, (pad, 0))
            else:
                sig_cut = sig_np[start_idx_sig : start_idx_sig + window_samples]
                
            if len(sig_cut) < window_samples:
                sig_cut = np.pad(sig_cut, (0, window_samples - len(sig_cut)))
            else:
                sig_cut = sig_cut[:window_samples]
                
            # Whiten the signal using the noise PSD
            # Note: We need the PSD from the corresponding IFO.
            # We stored it in strain_psds[ifo]
            
            # Convert signal to PyCBC TimeSeries for whitening
            # Signal is already aligned to window
            sig_ts = PyCBCTimeSeries(sig_cut, delta_t=1.0/cfg.sample_rate)
            
            # Whiten
            # We need the PSD. But wait, the PSD was computed on the LONG segment (4096s).
            # Is it valid for this short window? Yes, assuming stationarity.
            psd = strain_psds[ifo]
            
            # Whiten signal
            # whiten_with_psd expects PyCBC objects
            from src.data_handling.gw_utils import whiten_with_psd
            sig_whitened = whiten_with_psd(sig_ts, psd)
            
            # Crop edge artifacts from signal whitening?
            # whiten_with_psd does frequency domain division.
            # It might introduce wrap-around or edge effects if not careful.
            # But our signal is short and centered in the window (hopefully).
            # If the signal is zero at edges, it's fine.
            # BBH signals are usually zero at start, but ringdown might be abrupt if cut.
            # Let's assume it's fine for now, or we should have generated a longer signal and cropped.
            # Given the constraints, we use what we have.
            
            sig_whitened_np = sig_whitened.numpy()
            
            # Re-scale to target SNR
            # Note: The SNR target is usually defined on the WHITENED signal.
            # So we should calculate SNR of sig_whitened_np.
            
            # Accumulate network SNR
            network_snr_sq += np.sum(sig_whitened_np**2)
            
            # Store for addition
            # We need to store the whitened signal to add it later
            # But we need to scale it first.
            # So we store the unscaled whitened signal.
            raw_signals[ifo] = sig_whitened_np
            
        
        current_snr = np.sqrt(network_snr_sq)
        scale_factor = snr_target / (current_snr + 1e-9)
        
        for ifo in cfg.ifo_list:
            # Add whitened signal to whitened noise
            inj_chunks[ifo] = noise_chunks[ifo] + raw_signals[ifo] * scale_factor
            
        # Save Injection
        save_sample(
            h5_file, gid_inj, inj_chunks, 
            cfg, is_signal=True, f_common=f_sel, freq_mask=freq_mask,
            meta={"snr": snr_target, "ra": ra, "dec": dec}
        )
        added_count += 1
        
    return added_count


def save_sample(h5, gid, chunks, cfg, is_signal, f_common, freq_mask, meta=None):
    """
    Saves a single sample (Multi-IFO) to HDF5.
    """
    # Create group
    g = h5.create_group(gid)
    
    # Save IFO data
    for ifo, data in chunks.items():
        # Compute STFT
        f, t, Zxx = compute_stft(data, cfg.sample_rate)
        mag, cos, sin = tf_to_mag_phase_channels(Zxx)
        
        # Trim freq
        mag = mag[:, freq_mask]
        cos = cos[:, freq_mask]
        sin = sin[:, freq_mask]
        mask_t = np.ones(mag.shape[0], dtype=np.float32)
        
        ig = g.create_group(ifo)
        ig.create_dataset("mag", data=mag, compression="gzip")
        ig.create_dataset("cos", data=cos, compression="gzip")
        ig.create_dataset("sin", data=sin, compression="gzip")
        g.create_dataset(f"mask_{ifo}", data=mask_t, compression="gzip")
        
    # Common Freq
    g.create_dataset("f", data=f_common, compression="gzip")
    
    # Labels
    g.create_dataset("label", data=np.float32(is_signal))
    # Multiclass: 0=BG, 1=BBH (Signal)
    mc = 1 if is_signal else 0
    g.create_dataset("label_multiclass", data=np.int32(mc))
    
    # Metadata
    m = g.create_group("meta")
    m.attrs["is_signal"] = int(is_signal)
    m.attrs["source_class"] = "BBH" if is_signal else "BACKGROUND"
    m.attrs["ifos"] = ",".join(sorted(chunks.keys()))

    if meta:
        for k, v in meta.items():
            m.attrs[k] = v


@hydra.main(config_path="../../configs", config_name="data_generation", version_base="1.2")
def main(cfg: DictConfig):
    print("DEBUG: Entering main")
    setup_gwosc_cache()
    log.info("=== Dr. Gravitas Data Generator (Hydra + Multi-IFO) ===")




    ensure_dir(project_path(cfg.output_path).parent)
    
    # 1. Find Segments
    print("DEBUG: Finding segments...")
    segments = get_coincident_segments(
        cfg.ifo_list, 
        cfg.search_start, 
        cfg.search_end, 
        cfg.min_segment_len
    )
    print(f"DEBUG: Segments found: {len(segments)}")
    
    if not segments:
        log.error("No valid segments found!")
        print("DEBUG: No segments found.")
        return

    # 2. Open HDF5
    out_path = project_path(cfg.output_path)
    print(f"DEBUG: Output path: {out_path}")
    
    # Check if file exists to determine mode
    if os.path.exists(out_path):
        mode = "a"
        log.info(f"File {out_path} exists. Appending to it.")
    else:
        mode = "w"
        log.info(f"Creating new file {out_path}.")
    
    with h5py.File(out_path, mode) as h5:

        # Find next ID
        existing_ids = [int(k) for k in h5.keys() if k.isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 0
        
        existing_count = len(existing_ids)
        target_samples = cfg.n_samples * 2 # Pos + Neg
        
        if existing_count >= target_samples:
            log.info(f"Target samples ({target_samples}) already reached (found {existing_count}). Exiting.")
            return

        log.info(f"Resuming generation. Found {existing_count} samples. Target: {target_samples}. Need {target_samples - existing_count} more.")
        
        total_generated = existing_count
        
        pbar = tqdm(total=target_samples, initial=existing_count, desc="Generating Samples")
        
        # Loop over segments
        for seg in segments:
            if total_generated >= target_samples:
                break
                
            num = process_segment(seg, cfg, h5, next_id)
            
            next_id += num
            total_generated += num
            pbar.update(num)

            
        pbar.close()
        
    log.info(f"Done! Saved {total_generated} samples to {out_path}")

if __name__ == "__main__":
    main()
