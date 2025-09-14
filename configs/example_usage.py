#!/usr/bin/env python3
"""
Example usage of CPC-SNN-GW configuration system.

This demonstrates how to use the parameterized configuration
instead of hardcoded values throughout the codebase.
"""

from utils.config_loader import load_config, get_config_value

def main():
    """Demonstrate configuration system usage."""
    
    print("🔧 CPC-SNN-GW Configuration System Examples")
    print("=" * 50)
    
    # 1. Load full configuration
    print("\n1️⃣ Loading full configuration:")
    config = load_config()
    print(f"✅ Configuration loaded with {len(config)} sections")
    
    # 2. Access nested values
    print("\n2️⃣ Accessing configuration values:")
    data_dir = config['system']['data_dir']
    sample_rate = config['data']['sample_rate']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    
    print(f"   Data directory: {data_dir}")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # 3. Using convenience function
    print("\n3️⃣ Using convenience functions:")
    num_epochs = get_config_value('training.num_epochs', default=50)
    device = get_config_value('system.device', default='cpu')
    
    print(f"   Number of epochs: {num_epochs}")
    print(f"   Device: {device}")
    
    # 4. Environment variable overrides
    print("\n4️⃣ Environment variable examples:")
    print("   Set environment variables to override config:")
    print("   export CPC_SNN_BATCH_SIZE=16")
    print("   export CPC_SNN_LEARNING_RATE=0.001")
    print("   export CPC_SNN_DATA_DIR=/path/to/data")
    print("   export CPC_SNN_DEVICE=gpu")
    
    # 5. Custom configuration files
    print("\n5️⃣ Custom configuration files:")
    print("   Create configs/experiment.yaml for specific experiments")
    print("   Create configs/user.yaml for personal settings")
    print("   Use: config = load_config(experiment_config='experiment')")
    
    # 6. Component usage examples
    print("\n6️⃣ Component usage examples:")
    print("   # MLGWSC Data Loader")
    print("   from data.mlgwsc_data_loader import MLGWSCDataLoader")
    print("   loader = MLGWSCDataLoader()  # Uses config automatically")
    print()
    print("   # Training with config")
    print("   config = load_config()")
    print("   trainer = create_trainer(")
    print("       batch_size=config['training']['batch_size'],")
    print("       learning_rate=config['training']['learning_rate']")
    print("   )")
    
    print("\n✅ Configuration system ready for use!")
    print("   All hardcoded values have been parameterized.")
    print("   Modify configs/default.yaml to change defaults.")

if __name__ == "__main__":
    main()
