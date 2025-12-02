import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from gwsnr import GWSNR

def generate_balanced_bank(output_path, n_per_bin=100, snr_min=4.0, snr_max=20.0, sample_rate=4096.0, duration=4.0):
    """
    Generates a bank of signals balanced across SNR bins using gwsnr for SNR calculation.
    """
    print(f"Generating Balanced Signal Bank ({n_per_bin} per bin, SNR {snr_min}-{snr_max})...")
    
    # SNR Bins: [4, 5), [5, 6), ..., [19, 20)
    bins = np.arange(snr_min, snr_max + 1.0, 1.0)
    bank = {f"{int(b)}-{int(b+1)}": [] for b in bins[:-1]}
    
    # Initialize GWSNR
    # We use IMRPhenomD to match pycbc generation
    gwsnr = GWSNR(
        npool=1, 
        snr_method='interpolation_no_spins', # Fast interpolation
        snr_type='optimal_snr',
        waveform_approximant='IMRPhenomD',
        sampling_frequency=sample_rate,
        minimum_frequency=20.0,
        ifos=['H1', 'L1']
    )
    
    target_samples = int(duration * sample_rate)
    
    pbar = tqdm(total=len(bank) * n_per_bin)
    
    while any(len(v) < n_per_bin for v in bank.values()):
        # Generate random parameters
        # gwsnr expects arrays
        m1 = np.random.uniform(10, 50)
        m2 = np.random.uniform(10, 50)
        dist = np.random.uniform(100, 2000) # Mpc
        
        ra = np.random.uniform(0, 2 * np.pi)
        dec = np.random.uniform(-np.pi / 2, np.pi / 2)
        psi = np.random.uniform(0, 2 * np.pi)
        theta_jn = np.random.uniform(0, np.pi) # Inclination
        # pycbc get_td_waveform uses 'inclination', gwsnr uses 'theta_jn' (same thing usually)
        
        # Calculate SNR using gwsnr
        # optimal_snr returns dict with 'optimal_snr_net'
        snr_dict = gwsnr.optimal_snr(
            mass_1=m1,
            mass_2=m2,
            luminosity_distance=dist,
            theta_jn=theta_jn,
            psi=psi,
            ra=ra,
            dec=dec,
            geocent_time=1000000000.0 # Fixed time
        )
        
        net_snr = snr_dict['snr_net'][0] # returns array
        
        # Check bin
        bin_idx = int(net_snr)
        bin_key = f"{bin_idx}-{bin_idx+1}"
        
        if bin_key in bank and len(bank[bin_key]) < n_per_bin:
            # Generate Waveform using PyCBC (to get the time series)
            # We must use SAME parameters
            hp, hc = get_td_waveform(approximant="IMRPhenomD",
                                     mass1=m1,
                                     mass2=m2,
                                     distance=dist,
                                     inclination=theta_jn,
                                     coa_phase=0.0, # gwsnr default phase=0
                                     f_lower=20.0,
                                     delta_t=1.0/sample_rate)
            
            # Project to detectors
            t_gps = 1000000000.0
            det_h1 = Detector("H1")
            det_l1 = Detector("L1")
            
            fp_h1, fc_h1 = det_h1.antenna_pattern(ra, dec, psi, t_gps)
            fp_l1, fc_l1 = det_l1.antenna_pattern(ra, dec, psi, t_gps)
            
            s_h1 = fp_h1 * hp + fc_h1 * hc
            s_l1 = fp_l1 * hp + fc_l1 * hc
            
            # Process (Crop/Pad)
            sig_channels = []
            for sig in [s_h1, s_l1]:
                sig_np = sig.numpy()
                peak_idx = np.argmax(np.abs(sig_np))
                start = peak_idx - target_samples // 2
                end = start + target_samples
                
                if start < 0:
                    pad_left = -start
                    sig_cut = np.pad(sig_np, (pad_left, 0))[:target_samples]
                elif end > len(sig_np):
                    pad_right = end - len(sig_np)
                    sig_cut = np.pad(sig_np, (0, pad_right))[start:]
                else:
                    sig_cut = sig_np[start:end]
                    
                if len(sig_cut) < target_samples:
                     sig_cut = np.pad(sig_cut, (0, target_samples - len(sig_cut)))
                
                sig_channels.append(sig_cut)
            
            # Stack: (2, T)
            sig_tensor = torch.tensor(np.stack(sig_channels), dtype=torch.float32)
            bank[bin_key].append(sig_tensor)
            pbar.update(1)
            
    pbar.close()
    
    # Save
    print(f"Saving bank to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(bank, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/balanced_signal_bank.pt")
    parser.add_argument("--n_per_bin", type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_balanced_bank(args.output, n_per_bin=args.n_per_bin)
