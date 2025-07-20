"""
Configuration management for LIGO CPC+SNN Pipeline

Centralized configuration loading and validation using dataclasses.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import logging

from ..models.spike_bridge import SpikeEncodingStrategy

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """Platform and hardware configuration."""
    device: str = "auto"  # auto, metal, cpu, gpu
    precision: str = "float32"
    enable_x64: bool = False
    
    def __post_init__(self):
        """Auto-detect device if set to 'auto'."""
        if self.device == "auto":
            import jax
            # Try to detect available devices
            try:
                devices = jax.devices()
                if any("gpu" in str(device) for device in devices):
                    self.device = "gpu"
                elif any("metal" in str(device) for device in devices):
                    self.device = "metal"
                else:
                    self.device = "cpu"
            except:
                self.device = "cpu"  # Fallback to CPU


@dataclass
class DataConfig:
    """Data configuration."""
    sample_rate: int = 16384  # Hz
    segment_duration: float = 4.0  # seconds
    detectors: List[str] = field(default_factory=lambda: ["H1", "L1"])
    preprocessing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPCConfig:
    """CPC encoder configuration."""
    latent_dim: int = 256
    downsample_factor: int = 64
    context_length: int = 12
    num_negatives: int = 16
    temperature: float = 0.1
    architecture: Dict[str, Any] = field(default_factory=lambda: {
        "conv_channels": [32, 64, 128, 256],
        "conv_kernel_sizes": [7, 5, 3, 3],
        "conv_strides": [2, 2, 2, 2],
        "dropout_rate": 0.1,
        "use_batch_norm": True
    })


@dataclass
class SpikeBridgeConfig:
    """Spike bridge configuration."""
    encoding_strategy: SpikeEncodingStrategy = SpikeEncodingStrategy.POISSON_RATE
    spike_time_steps: int = 100
    max_spike_rate: float = 100.0  # Hz
    dt: float = 0.001
    population_size: int = 8


@dataclass
class SNNConfig:
    """SNN configuration."""
    hidden_size: int = 128
    num_layers: int = 2
    neuron_type: str = "LIF"
    tau_mem: float = 0.020  # seconds
    tau_syn: float = 0.005  # seconds
    threshold: float = 1.0
    num_classes: int = 2


@dataclass
class TrainingPhaseConfig:
    """Training phase configuration."""
    steps: int = 10000
    batch_size: int = 16
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    grad_clip: float = 1.0
    
    def __post_init__(self):
        """Validate grad_clip to prevent division by zero."""
        if self.grad_clip <= 0:
            logger.warning(f"grad_clip value {self.grad_clip} is invalid, setting to 1.0")
            self.grad_clip = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    cpc_pretrain: TrainingPhaseConfig = field(default_factory=TrainingPhaseConfig)
    snn_train: TrainingPhaseConfig = field(default_factory=TrainingPhaseConfig)
    joint_finetune: TrainingPhaseConfig = field(default_factory=TrainingPhaseConfig)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: ["roc_auc", "precision", "recall"])
    target_far: float = 1e-3  # False alarm rate
    target_tpr: float = 0.95  # True positive rate


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    wandb_project: str = "cpc-snn-gw"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cpc: CPCConfig = field(default_factory=CPCConfig)
    spike_bridge: SpikeBridgeConfig = field(default_factory=SpikeBridgeConfig)
    snn: SNNConfig = field(default_factory=SNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Validate configuration consistency."""
        # Validate num_classes consistency
        # For GW detection: 0=noise, 1=continuous, 2=binary (3 classes)
        expected_num_classes = 3
        if self.snn.num_classes != expected_num_classes:
            logger.warning(f"SNN num_classes {self.snn.num_classes} != expected {expected_num_classes}, adjusting")
            self.snn.num_classes = expected_num_classes


def load_config(config_path: Optional[Path] = None) -> ExperimentConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.yaml
        
    Returns:
        ExperimentConfig: Loaded configuration
    """
    if config_path is None:
        # Default to config.yaml in the package directory
        config_path = Path(__file__).parent.parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return ExperimentConfig()
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle spike bridge encoding strategy conversion
        spike_bridge_dict = config_dict.get("spike_bridge", {})
        if "encoding_strategy" in spike_bridge_dict:
            # Convert string to enum if needed
            encoding_str = spike_bridge_dict["encoding_strategy"]
            if isinstance(encoding_str, str):
                try:
                    spike_bridge_dict["encoding_strategy"] = SpikeEncodingStrategy(encoding_str)
                except ValueError:
                    logger.warning(f"Unknown encoding strategy: {encoding_str}, using default")
                    spike_bridge_dict["encoding_strategy"] = SpikeEncodingStrategy.POISSON_RATE
        
        # Legacy support: convert "encoding" to "encoding_strategy"
        if "encoding" in spike_bridge_dict:
            encoding_map = {
                "poisson": SpikeEncodingStrategy.POISSON_RATE,
                "temporal_contrast": SpikeEncodingStrategy.TEMPORAL_CONTRAST,
                "population_vector": SpikeEncodingStrategy.POPULATION_VECTOR,
                "rate_based": SpikeEncodingStrategy.RATE_BASED
            }
            encoding_str = spike_bridge_dict.pop("encoding")
            if encoding_str in encoding_map:
                spike_bridge_dict["encoding_strategy"] = encoding_map[encoding_str]
                logger.info(f"Converted legacy encoding '{encoding_str}' to encoding_strategy")
            else:
                logger.warning(f"Unknown legacy encoding: {encoding_str}, using default")
        
        # Create configuration objects from dict
        config = ExperimentConfig(
            platform=PlatformConfig(**config_dict.get("platform", {})),
            data=DataConfig(**config_dict.get("data", {})),
            cpc=CPCConfig(**config_dict.get("cpc", {})),
            spike_bridge=SpikeBridgeConfig(**spike_bridge_dict),
            snn=SNNConfig(**config_dict.get("snn", {})),
            training=TrainingConfig(
                cpc_pretrain=TrainingPhaseConfig(**config_dict.get("training", {}).get("cpc_pretrain", {})),
                snn_train=TrainingPhaseConfig(**config_dict.get("training", {}).get("snn_train", {})),
                joint_finetune=TrainingPhaseConfig(**config_dict.get("training", {}).get("joint_finetune", {}))
            ),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            logging=LoggingConfig(**config_dict.get("logging", {}))
        )
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return ExperimentConfig()


def save_config(config: ExperimentConfig, config_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclasses to dict
    config_dict = {
        "platform": {
            "device": config.platform.device,
            "precision": config.platform.precision,
            "enable_x64": config.platform.enable_x64
        },
        "data": {
            "sample_rate": config.data.sample_rate,
            "segment_duration": config.data.segment_duration,
            "detectors": config.data.detectors,
            "preprocessing": config.data.preprocessing
        },
        "cpc": {
            "latent_dim": config.cpc.latent_dim,
            "downsample_factor": config.cpc.downsample_factor,
            "context_length": config.cpc.context_length,
            "num_negatives": config.cpc.num_negatives,
            "temperature": config.cpc.temperature,
            "architecture": config.cpc.architecture
        },
        "spike_bridge": {
            "encoding_strategy": config.spike_bridge.encoding_strategy.value,  # Convert enum to string
            "spike_time_steps": config.spike_bridge.spike_time_steps,
            "max_spike_rate": config.spike_bridge.max_spike_rate,
            "dt": config.spike_bridge.dt,
            "population_size": config.spike_bridge.population_size
        },
        "snn": {
            "hidden_size": config.snn.hidden_size,
            "num_layers": config.snn.num_layers,
            "neuron_type": config.snn.neuron_type,
            "tau_mem": config.snn.tau_mem,
            "tau_syn": config.snn.tau_syn,
            "threshold": config.snn.threshold,
            "num_classes": config.snn.num_classes
        },
        "training": {
            "cpc_pretrain": {
                "steps": config.training.cpc_pretrain.steps,
                "batch_size": config.training.cpc_pretrain.batch_size,
                "learning_rate": config.training.cpc_pretrain.learning_rate,
                "optimizer": config.training.cpc_pretrain.optimizer,
                "grad_clip": config.training.cpc_pretrain.grad_clip
            },
            "snn_train": {
                "steps": config.training.snn_train.steps,
                "batch_size": config.training.snn_train.batch_size,
                "learning_rate": config.training.snn_train.learning_rate,
                "optimizer": config.training.snn_train.optimizer,
                "grad_clip": config.training.snn_train.grad_clip
            },
            "joint_finetune": {
                "steps": config.training.joint_finetune.steps,
                "batch_size": config.training.joint_finetune.batch_size,
                "learning_rate": config.training.joint_finetune.learning_rate,
                "optimizer": config.training.joint_finetune.optimizer,
                "grad_clip": config.training.joint_finetune.grad_clip
            }
        },
        "evaluation": {
            "metrics": config.evaluation.metrics,
            "target_far": config.evaluation.target_far,
            "target_tpr": config.evaluation.target_tpr
        },
        "logging": {
            "level": config.logging.level,
            "log_dir": config.logging.log_dir,
            "wandb_project": config.logging.wandb_project,
            "save_checkpoints": config.logging.save_checkpoints,
            "checkpoint_dir": config.logging.checkpoint_dir
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration saved to {config_path}")


def validate_config(config: ExperimentConfig) -> bool:
    """
    Validate configuration for consistency and correctness.
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Platform validation
        assert config.platform.device in ["metal", "cpu", "gpu"], f"Invalid device: {config.platform.device}"
        assert config.platform.precision in ["float32", "float16"], f"Invalid precision: {config.platform.precision}"
        
        # Data validation
        assert config.data.sample_rate > 0, "Sample rate must be positive"
        assert config.data.segment_duration > 0, "Segment duration must be positive"
        assert len(config.data.detectors) > 0, "At least one detector must be specified"
        
        # CPC validation
        assert config.cpc.latent_dim > 0, "Latent dimension must be positive"
        assert config.cpc.downsample_factor > 0, "Downsample factor must be positive"
        assert config.cpc.context_length > 0, "Context length must be positive"
        assert config.cpc.num_negatives > 0, "Number of negatives must be positive"
        assert config.cpc.temperature > 0, "Temperature must be positive"
        
        # Spike bridge validation
        assert isinstance(config.spike_bridge.encoding_strategy, SpikeEncodingStrategy), "Encoding strategy must be SpikeEncodingStrategy enum"
        assert config.spike_bridge.spike_time_steps > 0, "Spike time steps must be positive"
        assert config.spike_bridge.max_spike_rate > 0, "Max spike rate must be positive"
        assert config.spike_bridge.dt > 0, "Time step must be positive"
        assert config.spike_bridge.population_size > 0, "Population size must be positive"
        
        # SNN validation
        assert config.snn.hidden_size > 0, "Hidden size must be positive"
        assert config.snn.num_layers > 0, "Number of layers must be positive"
        assert config.snn.neuron_type in ["LIF", "IF"], f"Invalid neuron type: {config.snn.neuron_type}"
        assert config.snn.tau_mem > 0, "Membrane time constant must be positive"
        assert config.snn.tau_syn > 0, "Synaptic time constant must be positive"
        assert config.snn.threshold > 0, "Threshold must be positive"
        assert config.snn.num_classes > 0, "Number of classes must be positive"
        
        # Training validation
        for phase_name, phase_config in [
            ("cpc_pretrain", config.training.cpc_pretrain),
            ("snn_train", config.training.snn_train),
            ("joint_finetune", config.training.joint_finetune)
        ]:
            assert phase_config.steps > 0, f"{phase_name} steps must be positive"
            assert phase_config.batch_size > 0, f"{phase_name} batch size must be positive"
            assert phase_config.learning_rate > 0, f"{phase_name} learning rate must be positive"
            assert phase_config.optimizer in ["adam", "sgd", "rmsprop", "adamw"], f"Invalid optimizer: {phase_config.optimizer}"
            assert phase_config.grad_clip > 0, f"{phase_name} gradient clipping must be positive"
        
        # Evaluation validation
        valid_metrics = ["roc_auc", "precision", "recall", "f1", "accuracy"]
        for metric in config.evaluation.metrics:
            assert metric in valid_metrics, f"Invalid metric: {metric}"
        assert 0 < config.evaluation.target_far < 1, "Target FAR must be between 0 and 1"
        assert 0 < config.evaluation.target_tpr < 1, "Target TPR must be between 0 and 1"
        
        logger.info("Configuration validation passed")
        return True
        
    except AssertionError as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return False 