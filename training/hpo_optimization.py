"""
Systematic Hyperparameter Optimization Framework for LIGO CPC+SNN

Implements systematic hyperparameter search as recommended in Executive Summary.
Critical for maximizing performance and strengthening scientific value of results.
"""

import jax
import jax.numpy as jnp
import optuna
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pickle

from .advanced_training import AdvancedTrainingConfig, AdvancedGWTrainer
from .training_utils import setup_training_environment
from ..utils.config import apply_performance_optimizations

logger = logging.getLogger(__name__)

@dataclass 
class HPOConfiguration:
    """Configuration for hyperparameter optimization experiments"""
    
    # Optimization settings
    study_name: str = "ligo_cpc_snn_hpo"
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    
    # Search space limits
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_options: List[int] = None
    cpc_latent_dim_options: List[int] = None
    snn_hidden_sizes_options: List[Tuple[int, ...]] = None
    
    # Training limits for HPO
    max_epochs_per_trial: int = 20  # Reduced for HPO speed
    early_stopping_patience: int = 5
    min_accuracy_threshold: float = 0.6  # Early pruning
    
    # Hardware optimization
    use_distributed: bool = False
    max_parallel_trials: int = 4
    
    # Results storage
    results_dir: str = "hpo_results"
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [16, 32, 64, 128]
        
        if self.cpc_latent_dim_options is None:
            self.cpc_latent_dim_options = [128, 256, 512]
            
        if self.snn_hidden_sizes_options is None:
            self.snn_hidden_sizes_options = [
                (128, 64),      # Shallow
                (256, 128, 64), # Standard  
                (512, 256, 128, 64),  # Deep
                (256, 256, 128),      # Wide
                (128, 128, 128, 64)   # Uniform
            ]

class HPOSearchSpace:
    """Defines search space for hyperparameter optimization"""
    
    @staticmethod
    def suggest_hyperparameters(trial: optuna.Trial, 
                               hpo_config: HPOConfiguration) -> AdvancedTrainingConfig:
        """
        Suggest hyperparameters for a single trial
        
        Args:
            trial: Optuna trial object
            hpo_config: HPO configuration
            
        Returns:
            AdvancedTrainingConfig with suggested hyperparameters
        """
        
        # Core optimization parameters
        learning_rate = trial.suggest_float(
            'learning_rate', 
            hpo_config.learning_rate_range[0],
            hpo_config.learning_rate_range[1],
            log=True
        )
        
        batch_size = trial.suggest_categorical(
            'batch_size',
            hpo_config.batch_size_options
        )
        
        # Model architecture parameters
        cpc_latent_dim = trial.suggest_categorical(
            'cpc_latent_dim',
            hpo_config.cpc_latent_dim_options
        )
        
        snn_hidden_sizes = trial.suggest_categorical(
            'snn_hidden_sizes',
            hpo_config.snn_hidden_sizes_options
        )
        
        # Training technique parameters
        use_attention = trial.suggest_categorical(
            'use_attention', [True, False]
        )
        
        use_focal_loss = trial.suggest_categorical(
            'use_focal_loss', [True, False]
        )
        
        use_mixup = trial.suggest_categorical(
            'use_mixup', [True, False]
        )
        
        # Focal loss parameters (if enabled)
        focal_alpha = trial.suggest_float(
            'focal_alpha', 0.1, 0.9
        ) if use_focal_loss else 0.25
        
        focal_gamma = trial.suggest_float(
            'focal_gamma', 1.0, 5.0
        ) if use_focal_loss else 2.0
        
        # Mixup parameters (if enabled)
        mixup_alpha = trial.suggest_float(
            'mixup_alpha', 0.1, 0.5
        ) if use_mixup else 0.2
        
        # Regularization parameters
        weight_decay = trial.suggest_float(
            'weight_decay', 1e-6, 1e-2, log=True
        )
        
        dropout_rate = trial.suggest_float(
            'dropout_rate', 0.0, 0.5
        )
        
        # Learning rate scheduling
        use_cosine_scheduling = trial.suggest_categorical(
            'use_cosine_scheduling', [True, False]
        )
        
        warmup_epochs = trial.suggest_int(
            'warmup_epochs', 5, 20
        ) if use_cosine_scheduling else 10
        
        # Spike encoding parameters
        spike_time_steps = trial.suggest_categorical(
            'spike_time_steps', [50, 100, 200]
        )
        
        # Create configuration
        config = AdvancedTrainingConfig(
            # Basic training
            num_epochs=hpo_config.max_epochs_per_trial,
            learning_rate=learning_rate,
            batch_size=batch_size,
            
            # Model architecture
            cpc_latent_dim=cpc_latent_dim,
            snn_hidden_sizes=snn_hidden_sizes,
            spike_time_steps=spike_time_steps,
            
            # Advanced techniques
            use_attention=use_attention,
            use_focal_loss=use_focal_loss,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            
            # Regularization
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            
            # Scheduling
            use_cosine_scheduling=use_cosine_scheduling,
            warmup_epochs=warmup_epochs,
            
            # Output
            output_dir=f"hpo_trial_{trial.number}"
        )
        
        return config

class HPOObjective:
    """Objective function for hyperparameter optimization"""
    
    def __init__(self, hpo_config: HPOConfiguration):
        self.hpo_config = hpo_config
        self.setup_logging()
        
        # Setup training environment once
        setup_training_environment()
        apply_performance_optimizations()
    
    def setup_logging(self):
        """Setup logging for HPO trials"""
        log_dir = Path(self.hpo_config.results_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - HPO - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "hpo.log"),
                logging.StreamHandler()
            ]
        )
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function for single HPO trial
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (higher is better)
        """
        try:
            # Get hyperparameters for this trial
            config = HPOSearchSpace.suggest_hyperparameters(trial, self.hpo_config)
            
            logger.info(f"🔬 Starting HPO Trial {trial.number}")
            logger.info(f"Parameters: {asdict(config)}")
            
            # Create and run trainer
            trainer = AdvancedGWTrainer(config)
            
            # Run training with early stopping
            results = self._run_training_trial(trainer, trial)
            
            # Extract objective value
            objective_value = results['best_val_accuracy']
            
            # Early pruning if performance is poor
            if objective_value < self.hpo_config.min_accuracy_threshold:
                logger.info(f"⚡ Trial {trial.number} pruned: accuracy {objective_value:.3f} < {self.hpo_config.min_accuracy_threshold}")
                raise optuna.TrialPruned()
            
            # Save trial results
            if self.hpo_config.save_intermediate_results:
                self._save_trial_results(trial, results, config)
            
            logger.info(f"✅ Trial {trial.number} completed: accuracy = {objective_value:.4f}")
            
            return objective_value
            
        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {str(e)}")
            # Return poor score instead of crashing
            return 0.0
    
    def _run_training_trial(self, trainer: AdvancedGWTrainer, trial: optuna.Trial) -> Dict[str, Any]:
        """Run training for a single trial with early stopping"""
        
        best_val_accuracy = 0.0
        patience_counter = 0
        epoch_results = []
        
        # 🚨 CRITICAL FIX: Real training loop (not mock simulation)
        logger.info("🚀 Starting REAL HPO training loop...")
        
        # Create real trainer with trial parameters
        from training.advanced_training import create_advanced_trainer, AdvancedTrainingConfig
        
        try:
            # Setup training config from trial parameters
            training_config = AdvancedTrainingConfig(
                learning_rate=trial.params.get('learning_rate', 1e-4),
                batch_size=trial.params.get('batch_size', 16),
                cpc_latent_dim=trial.params.get('cpc_latent_dim', 512),
                snn_hidden_sizes=trial.params.get('snn_hidden_sizes', [256, 128, 64]),
                weight_decay=trial.params.get('weight_decay', 0.01),
                use_attention=trial.params.get('use_attention', True),
                use_focal_loss=trial.params.get('use_focal_loss', True),
                num_epochs=self.hpo_config.max_epochs_per_trial,
                output_dir=f"hpo_trial_{trial.number}"
            )
            
            # Create and run real trainer
            trainer = create_advanced_trainer(training_config)
            
            # ✅ REAL TRAINING: Execute actual training with pruning
            for epoch in range(self.hpo_config.max_epochs_per_trial):
                
                # Run real training epoch
                epoch_result = trainer.train_single_epoch(epoch)
                
                # Extract real validation accuracy
                val_accuracy = epoch_result.get('val_accuracy', 0.0)
                train_loss = epoch_result.get('train_loss', float('inf'))
                
                # Report real intermediate value for pruning
                trial.report(val_accuracy, epoch)
                if trial.should_prune():
                    logger.info(f"   🔪 Trial pruned at epoch {epoch} (val_accuracy={val_accuracy:.3f})")
                    raise optuna.TrialPruned()
                
                # Real early stopping logic
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    logger.info(f"   ✅ Epoch {epoch}: New best accuracy {val_accuracy:.3f}")
                else:
                    patience_counter += 1
                    
                # Early stopping check
                if patience_counter >= self.hpo_config.patience:
                    logger.info(f"   🛑 Early stopping at epoch {epoch} (patience={self.hpo_config.patience})")
                    break
                    
                epoch_results.append({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_accuracy': val_accuracy,
                    'best_val_accuracy': best_val_accuracy
                })
                
        except optuna.TrialPruned:
            # Re-raise pruning for Optuna
            raise
        except Exception as training_error:
            logger.error(f"   ❌ Real training failed in HPO: {training_error}")
            # Use conservative estimate for failed trials
            best_val_accuracy = 0.3  # Below random performance to discourage bad configs
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'num_epochs_trained': len(epoch_results),
            'epoch_results': epoch_results
        }
    
    def _save_trial_results(self, trial: optuna.Trial, 
                           results: Dict[str, Any],
                           config: AdvancedTrainingConfig):
        """Save detailed results for a trial"""
        results_dir = Path(self.hpo_config.results_dir) / f"trial_{trial.number}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial parameters and results
        trial_data = {
            'trial_number': trial.number,
            'objective_value': results['best_val_accuracy'],
            'hyperparameters': asdict(config),
            'results': results,
            'timestamp': time.time()
        }
        
        with open(results_dir / "trial_results.json", 'w') as f:
            json.dump(trial_data, f, indent=2)

class HPORunner:
    """Main runner for hyperparameter optimization experiments"""
    
    def __init__(self, hpo_config: Optional[HPOConfiguration] = None):
        self.config = hpo_config or HPOConfiguration()
        self.setup_results_directory()
    
    def setup_results_directory(self):
        """Setup results directory structure"""
        results_path = Path(self.config.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # Save HPO configuration
        config_file = results_path / "hpo_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def run_optimization(self) -> optuna.Study:
        """
        Run complete hyperparameter optimization
        
        Returns:
            Completed Optuna study with results
        """
        logger.info("🚀 Starting Systematic Hyperparameter Optimization")
        logger.info(f"Configuration: {asdict(self.config)}")
        
        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction='maximize',  # Maximize accuracy
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        
        # Create objective function
        objective = HPOObjective(self.config)
        
        # Run optimization
        if self.config.use_distributed and self.config.max_parallel_trials > 1:
            # Parallel optimization
            logger.info(f"Running {self.config.max_parallel_trials} parallel trials")
            
            with ProcessPoolExecutor(max_workers=self.config.max_parallel_trials) as executor:
                study.optimize(
                    objective,
                    n_trials=self.config.n_trials,
                    timeout=self.config.timeout,
                    n_jobs=self.config.max_parallel_trials
                )
        else:
            # Sequential optimization
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout
            )
        
        # Save final results
        self._save_study_results(study)
        
        logger.info("🎉 Hyperparameter optimization completed!")
        self._print_best_results(study)
        
        return study
    
    def _save_study_results(self, study: optuna.Study):
        """Save complete study results"""
        results_path = Path(self.config.results_dir)
        
        # Save study object
        with open(results_path / "study.pkl", 'wb') as f:
            pickle.dump(study, f)
        
        # Save best parameters
        best_params = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial_number': study.best_trial.number
        }
        
        with open(results_path / "best_results.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save trials dataframe
        df = study.trials_dataframe()
        df.to_csv(results_path / "trials.csv", index=False)
        
        logger.info(f"💾 Results saved to {results_path}")
    
    def _print_best_results(self, study: optuna.Study):
        """Print summary of best results"""
        logger.info("=" * 50)
        logger.info("🏆 BEST HYPERPARAMETERS FOUND:")
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info("Best parameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)

# Factory functions and utilities
def create_hpo_runner(n_trials: int = 100,
                     max_epochs_per_trial: int = 20,
                     results_dir: str = "hpo_results") -> HPORunner:
    """Factory function to create HPO runner with common settings"""
    
    config = HPOConfiguration(
        n_trials=n_trials,
        max_epochs_per_trial=max_epochs_per_trial,
        results_dir=results_dir,
        early_stopping_patience=5,
        min_accuracy_threshold=0.6
    )
    
    return HPORunner(config)

def run_quick_hpo_experiment(n_trials: int = 20) -> optuna.Study:
    """Run quick HPO experiment for testing"""
    logger.info("🔬 Running Quick HPO Experiment")
    
    runner = create_hpo_runner(
        n_trials=n_trials,
        max_epochs_per_trial=10,
        results_dir="quick_hpo_results"
    )
    
    return runner.run_optimization()

def run_full_hpo_experiment(n_trials: int = 100) -> optuna.Study:
    """Run full systematic HPO experiment"""
    logger.info("🚀 Running Full Systematic HPO Experiment")
    
    runner = create_hpo_runner(
        n_trials=n_trials,
        max_epochs_per_trial=50,
        results_dir="full_hpo_results"
    )
    
    return runner.run_optimization()

if __name__ == "__main__":
    # Run HPO optimization
    import os
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    # Quick test
    study = run_quick_hpo_experiment(n_trials=5)
    print(f"✅ Quick HPO completed! Best accuracy: {study.best_value:.4f}")
    
    # Uncomment for full experiment
    # study = run_full_hpo_experiment(n_trials=100)
    # print(f"🎉 Full HPO completed! Best accuracy: {study.best_value:.4f}") 