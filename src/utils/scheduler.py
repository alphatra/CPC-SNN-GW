import numpy as np
import math

class SNRScheduler:
    """
    Dynamic SNR Scheduler for Curriculum Learning in SNNs.
    Implements a Cosine Decay schedule with a Warmup phase.
    
    Phase 1: Warmup (Imprinting) - High constant SNR to help SNN learn features.
    Phase 2: Decay (Adaptation) - Smoothly decay SNR to min_snr.
    """
    def __init__(self, start_snr: float = 20.0, min_snr: float = 4.0, total_epochs: int = 50, warmup_epochs: int = 5):
        self.start_snr = start_snr
        self.min_snr = min_snr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_snr(self, current_epoch: int) -> float:
        # Phase 1: Warmup
        if current_epoch < self.warmup_epochs:
            return self.start_snr
        
        # Phase 2: Decay
        # Progress from 0.0 to 1.0
        if current_epoch >= self.total_epochs:
             return self.min_snr

        progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        # Cosine Decay: 0.5 * (1 + cos(pi * progress)) goes from 1.0 to 0.0
        decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        current_snr = self.min_snr + (self.start_snr - self.min_snr) * decay
        return current_snr
