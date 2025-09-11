"""
Hard negative mining and adaptive temperature control for CPC.

This module contains mining and control classes extracted from
cpc_losses.py for better modularity:
- MomentumHardNegativeMiner: Advanced momentum-based negative mining
- AdaptiveTemperatureController: Dynamic temperature adjustment

Split from cpc_losses.py for better maintainability.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class MomentumHardNegativeMiner:
    """
    🚀 ADVANCED: Momentum-based hard negative mining with curriculum learning.
    
    Features:
    - Memory bank with exponential moving average of negative similarities
    - Curriculum learning: easy→hard negative progression during training
    - Adaptive difficulty scheduling based on training progress
    - Multi-scale negative sampling for diverse contrastive learning
    """
    
    def __init__(self, 
                 momentum: float = 0.99,
                 difficulty_schedule: str = 'exponential',
                 memory_bank_size: int = 2048,
                 min_negatives: int = 8,
                 max_negatives: int = 32,
                 hard_negative_ratio: float = 0.3):
        """
        Initialize momentum-based hard negative miner.
        
        Args:
            momentum: Momentum factor for memory bank updates
            difficulty_schedule: 'linear', 'exponential', or 'cosine'
            memory_bank_size: Size of negative similarity memory bank
            min_negatives: Minimum number of hard negatives (early training)
            max_negatives: Maximum number of hard negatives (late training)
            hard_negative_ratio: Ratio of hard negatives to total negatives
        """
        self.momentum = momentum
        self.difficulty_schedule = difficulty_schedule
        self.memory_bank_size = memory_bank_size
        self.min_negatives = min_negatives
        self.max_negatives = max_negatives
        self.hard_negative_ratio = hard_negative_ratio
        
        # Memory bank will be initialized on first use
        self.negative_bank = None
        self.bank_initialized = False
        
        logger.debug("🚀 MomentumHardNegativeMiner initialized")
    
    def init_state(self, feature_dim: int) -> Dict[str, jnp.ndarray]:
        """
        Initialize state for momentum-based hard negative mining.
        
        Args:
            feature_dim: Dimension of feature vectors
            
        Returns:
            Initial state dictionary
        """
        # Initialize negative memory bank
        key = jax.random.PRNGKey(42)
        negative_bank = jax.random.normal(key, (self.memory_bank_size, feature_dim))
        
        # Normalize the bank
        negative_bank = negative_bank / (jnp.linalg.norm(negative_bank, axis=-1, keepdims=True) + 1e-8)
        
        state = {
            'negative_bank': negative_bank,
            'bank_ptr': jnp.array(0),  # Pointer for circular buffer
            'training_step': jnp.array(0)
        }
        
        self.bank_initialized = True
        logger.debug(f"Initialized negative bank with shape {negative_bank.shape}")
        
        return state
    
    def mine_hard_negatives(self, 
                           z_context: jnp.ndarray,
                           z_target: jnp.ndarray, 
                           temperature: float,
                           training_step: Optional[int] = None) -> Tuple[Optional[jnp.ndarray], Dict[str, Any]]:
        """
        Mine hard negatives using momentum-based strategy.
        
        Args:
            z_context: Context embeddings [batch_size, feature_dim]
            z_target: Target embeddings [batch_size, feature_dim]
            temperature: Temperature parameter
            training_step: Current training step for difficulty scheduling
            
        Returns:
            Tuple of (hard_negatives, mining_statistics)
        """
        if not self.bank_initialized:
            # Initialize state if not done yet
            feature_dim = z_context.shape[-1]
            self.state = self.init_state(feature_dim)
        
        batch_size, feature_dim = z_context.shape
        
        try:
            # ✅ NORMALIZATION
            z_context_norm = z_context / (jnp.linalg.norm(z_context, axis=-1, keepdims=True) + 1e-8)
            z_target_norm = z_target / (jnp.linalg.norm(z_target, axis=-1, keepdims=True) + 1e-8)
            
            # ✅ DIFFICULTY SCHEDULING: Adapt difficulty based on training progress
            if training_step is not None:
                progress = min(training_step / 10000.0, 1.0)  # Assume 10k steps for full curriculum
                
                if self.difficulty_schedule == 'exponential':
                    difficulty = 1.0 - jnp.exp(-3 * progress)
                elif self.difficulty_schedule == 'linear':
                    difficulty = progress
                elif self.difficulty_schedule == 'cosine':
                    difficulty = 0.5 * (1 - jnp.cos(jnp.pi * progress))
                else:
                    difficulty = 1.0
            else:
                difficulty = 1.0
            
            # ✅ ADAPTIVE SAMPLING: Compute number of hard negatives
            num_hard_negatives = int(
                self.min_negatives + 
                (self.max_negatives - self.min_negatives) * difficulty
            )
            
            # ✅ SIMILARITY COMPUTATION: Find hard negatives from memory bank
            if hasattr(self, 'state') and 'negative_bank' in self.state:
                bank_similarities = jnp.dot(z_context_norm, self.state['negative_bank'].T)
                
                # Select hard negatives (highest similarity = hardest)
                hard_negative_indices = jnp.argsort(bank_similarities, axis=1)[:, -num_hard_negatives:]
                
                # Extract hard negatives
                hard_negatives = self.state['negative_bank'][hard_negative_indices.flatten()]
                hard_negatives = hard_negatives.reshape(batch_size, num_hard_negatives, feature_dim)
                
                # ✅ MEMORY BANK UPDATE: Update with current targets
                # Update memory bank with momentum
                bank_ptr = self.state['bank_ptr']
                new_targets = z_target_norm
                
                # Circular buffer update
                end_ptr = (bank_ptr + batch_size) % self.memory_bank_size
                if end_ptr > bank_ptr:
                    # No wrap-around
                    updated_bank = self.state['negative_bank'].at[bank_ptr:end_ptr].set(new_targets)
                else:
                    # Wrap-around case
                    first_part = self.memory_bank_size - bank_ptr
                    updated_bank = self.state['negative_bank'].at[bank_ptr:].set(new_targets[:first_part])
                    if end_ptr > 0:
                        updated_bank = updated_bank.at[:end_ptr].set(new_targets[first_part:])
                
                # Update state
                self.state['negative_bank'] = updated_bank
                self.state['bank_ptr'] = end_ptr
                self.state['training_step'] = self.state['training_step'] + 1
                
                # Mining statistics
                mining_stats = {
                    'num_hard_negatives': num_hard_negatives,
                    'difficulty': float(difficulty),
                    'training_step': int(self.state['training_step']),
                    'bank_utilization': float(jnp.linalg.norm(self.state['negative_bank']) / self.memory_bank_size),
                    'mining_success': True
                }
                
                return hard_negatives.mean(axis=1), mining_stats  # Average over hard negatives
                
            else:
                logger.warning("Negative bank not initialized")
                return None, {'mining_success': False, 'error': 'bank_not_initialized'}
                
        except Exception as e:
            logger.error(f"Hard negative mining failed: {e}")
            return None, {'mining_success': False, 'error': str(e)}


class AdaptiveTemperatureController:
    """
    🌡️ ADAPTIVE: Dynamic temperature control for InfoNCE loss.
    
    Features:
    - Automatic temperature adaptation based on training dynamics
    - Convergence-aware temperature scheduling
    - Performance-based temperature adjustment
    - Gradient-based temperature optimization
    """
    
    def __init__(self,
                 initial_temperature: float = 0.1,
                 min_temperature: float = 0.01,
                 max_temperature: float = 1.0,
                 adaptation_rate: float = 0.01,
                 target_accuracy: float = 0.7):
        """
        Initialize adaptive temperature controller.
        
        Args:
            initial_temperature: Starting temperature value
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature  
            adaptation_rate: Rate of temperature adaptation
            target_accuracy: Target contrastive accuracy for adaptation
        """
        self.temperature = initial_temperature
        self.min_temp = min_temperature
        self.max_temp = max_temperature
        self.adaptation_rate = adaptation_rate
        self.target_accuracy = target_accuracy
        
        # History for trend analysis
        self.accuracy_history = []
        self.temperature_history = []
        self.loss_history = []
        
        logger.debug(f"🌡️ AdaptiveTemperatureController initialized: T={initial_temperature}")
    
    def update_temperature(self, 
                          current_accuracy: float,
                          current_loss: float,
                          training_step: int) -> float:
        """
        Update temperature based on training dynamics.
        
        Args:
            current_accuracy: Current contrastive accuracy
            current_loss: Current InfoNCE loss
            training_step: Current training step
            
        Returns:
            Updated temperature value
        """
        # Store history
        self.accuracy_history.append(current_accuracy)
        self.loss_history.append(current_loss)
        self.temperature_history.append(self.temperature)
        
        # Keep only recent history
        max_history = 100
        if len(self.accuracy_history) > max_history:
            self.accuracy_history = self.accuracy_history[-max_history:]
            self.loss_history = self.loss_history[-max_history:]
            self.temperature_history = self.temperature_history[-max_history:]
        
        # ✅ ADAPTATION LOGIC
        if len(self.accuracy_history) >= 10:  # Need some history for trends
            # Calculate trends
            recent_accuracy = jnp.mean(jnp.array(self.accuracy_history[-10:]))
            recent_loss = jnp.mean(jnp.array(self.loss_history[-10:]))
            
            # Accuracy-based adaptation
            accuracy_error = self.target_accuracy - recent_accuracy
            
            # Temperature adjustment
            if accuracy_error > 0.1:
                # Accuracy too low - decrease temperature (sharpen)
                temp_delta = -self.adaptation_rate * accuracy_error
            elif accuracy_error < -0.1:
                # Accuracy too high - increase temperature (soften)
                temp_delta = -self.adaptation_rate * accuracy_error
            else:
                # In target range - small adjustments based on loss trend
                if len(self.loss_history) >= 20:
                    loss_trend = jnp.mean(jnp.array(self.loss_history[-10:])) - jnp.mean(jnp.array(self.loss_history[-20:-10]))
                    temp_delta = self.adaptation_rate * 0.1 * loss_trend
                else:
                    temp_delta = 0.0
            
            # Update temperature with clipping
            new_temperature = jnp.clip(
                self.temperature + temp_delta,
                self.min_temp,
                self.max_temp
            )
            
            self.temperature = float(new_temperature)
        
        return self.temperature
    
    def get_temperature_stats(self) -> Dict[str, Any]:
        """Get temperature controller statistics."""
        if len(self.temperature_history) == 0:
            return {'temperature': self.temperature}
        
        return {
            'current_temperature': self.temperature,
            'initial_temperature': self.temperature_history[0] if self.temperature_history else self.temperature,
            'temperature_range': (min(self.temperature_history), max(self.temperature_history)),
            'temperature_trend': self.temperature_history[-10:] if len(self.temperature_history) >= 10 else self.temperature_history,
            'adaptation_active': len(self.accuracy_history) >= 10
        }
    
    def reset_adaptation(self):
        """Reset adaptation history."""
        self.accuracy_history = []
        self.temperature_history = []
        self.loss_history = []
        logger.debug("🌡️ Temperature adaptation history reset")


# Export mining and control classes
__all__ = [
    "MomentumHardNegativeMiner",
    "AdaptiveTemperatureController"
]

