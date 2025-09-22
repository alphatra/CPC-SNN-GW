#!/usr/bin/env python3
"""
test_cpc_fix.py

Szybki test sprawdzający czy naprawa normalizacji Z-score rozwiązała problem stagnacji cpc_loss.

Uruchamia krótki trening (5 epok) z naprawioną normalizacją i loguje wyniki.
"""

import sys
import os
sys.path.insert(0, '/teamspace/studios/this_studio/CPC-SNN-GW')

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Import z projektu
from training.base.trainer import CPCSNNTrainer
from training.base.config import TrainingConfig

def main():
    print("🔥 TESTOWANIE NAPRAWY CPC LOSS STAGNACJI")
    print("=" * 50)
    
    # Konfiguracja testowa
    config = TrainingConfig(
        model_name="test_cpc_fix",
        num_epochs=5,  # Krótki test
        batch_size=8,
        learning_rate=1e-4,  # Zgodnie z rekomendacjami
        output_dir="outputs/test_cpc_fix",
        
        # CPC parametry
        cpc_temperature=0.07,
        cpc_prediction_steps=4,
        
        # SNN parametry  
        spike_time_steps=32,
        spike_threshold=0.45,
        snn_hidden_sizes=(256, 128, 64),  # 3 warstwy jak w domyślnej konfiguracji
        snn_num_layers=3,
        snn_dropout_rates=(0.2, 0.1, 0.0),  # Dopasowane do liczby warstw
        
        # Inne
        early_stopping_patience=10,
        enable_profiling=False,
        use_wandb=False
    )
    
    # Stwórz dummy dane
    print("📊 Generowanie danych testowych...")
    np.random.seed(42)
    
    # Symulacja danych LIGO (1024 próbek na sygnał)
    num_samples = 200
    sequence_length = 1024
    
    # 50% noise, 50% sygnał + noise
    train_signals = []
    train_labels = []
    
    for i in range(num_samples):
        if i < num_samples // 2:
            # Pure noise
            signal = np.random.normal(0, 1, sequence_length)
            label = 0
        else:
            # Signal + noise (symulacja GW)
            t = np.linspace(0, 1, sequence_length)
            gw_signal = 0.1 * np.sin(2 * np.pi * 100 * t) * np.exp(-5 * t)  # Chirp-like
            noise = np.random.normal(0, 1, sequence_length)
            signal = gw_signal + noise
            label = 1
            
        train_signals.append(signal)
        train_labels.append(label)
    
    train_signals = jnp.array(train_signals, dtype=jnp.float32)
    # CPC oczekuje wejścia 3D: [batch, sequence, features]
    train_signals = train_signals[..., None]
    train_labels = jnp.array(train_labels, dtype=jnp.int32)
    
    # Test set (mniejszy)
    test_signals = train_signals[:40]
    test_labels = train_labels[:40]
    
    print(f"✅ Dane przygotowane: {len(train_signals)} train, {len(test_signals)} test")
    print(f"   Klasy: {np.bincount(train_labels)} (train), {np.bincount(test_labels)} (test)")
    
    # Inicjalizacja trainera
    print("🤖 Inicjalizacja trainera z naprawioną normalizacją...")
    trainer = CPCSNNTrainer(config)
    
    # Uruchomienie treningu
    print("🚀 Rozpoczynanie treningu...")
    print("⚠️  Obserwuj czy cpc_loss spada poniżej ~7.6!")
    print("-" * 50)
    
    try:
        results = trainer.train(
            train_signals=train_signals,
            train_labels=train_labels,
            test_signals=test_signals,
            test_labels=test_labels
        )
        
        print("-" * 50)
        print("🎊 TRENING ZAKOŃCZONY!")
        print(f"✅ Final accuracy: {results['test_accuracy']:.3f}")
        print(f"✅ Final loss: {results['test_loss']:.4f}")
        
        # Sprawdź logi JSONL
        log_dir = Path(config.output_dir) / "logs"
        step_log = log_dir / "training_results.jsonl"
        
        if step_log.exists():
            print(f"\n📊 Analiza logów z {step_log}:")
            with open(step_log, 'r') as f:
                lines = f.readlines()
                
            if lines:
                # Pierwsza i ostatnia linia
                import json
                first_step = json.loads(lines[0])
                last_step = json.loads(lines[-1])
                
                first_cpc = first_step.get('cpc_loss', 'N/A')
                last_cpc = last_step.get('cpc_loss', 'N/A')
                
                print(f"   🔍 Pierwsza cpc_loss: {first_cpc}")
                print(f"   🔍 Ostatnia cpc_loss: {last_cpc}")
                
                if isinstance(first_cpc, (int, float)) and isinstance(last_cpc, (int, float)):
                    if last_cpc < first_cpc:
                        print("   🎉 SUCCESS: CPC loss SPADA! Naprawa działa!")
                    else:
                        print("   ⚠️  WARNING: CPC loss nie spada znacząco")
                        
        else:
            print("⚠️  Brak logów JSONL")
            
    except Exception as e:
        print(f"❌ BŁĄD podczas treningu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
