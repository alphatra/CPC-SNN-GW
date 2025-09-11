"""
Advanced Training Module: Advanced Training Components

Modular implementation of advanced training components
split from advanced_training.py for better maintainability.

Components:
- attention: AttentionCPCEncoder for advanced CPC with attention
- snn_deep: DeepSNN and LIFLayer for deep spiking networks
- trainer: RealAdvancedGWTrainer for advanced GW detection training
"""

from .attention import AttentionCPCEncoder
from .snn_deep import DeepSNN, LIFLayer
from .trainer import RealAdvancedGWTrainer, create_real_advanced_trainer

__all__ = [
    # Attention components
    "AttentionCPCEncoder",
    
    # Deep SNN components
    "DeepSNN",
    "LIFLayer",
    
    # Advanced trainer
    "RealAdvancedGWTrainer",
    "create_real_advanced_trainer"
]
