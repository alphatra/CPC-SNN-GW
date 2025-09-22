"""
pretrain_cpc.py

Samodzielny skrypt do pre-treningu enkodera Contrastive Predictive Coding (CPC).

Cel:
Zweryfikować, czy enkoder CPC jest w stanie nauczyć się użytecznych reprezentacji
z danych sekwencyjnych. Spadek wartości `cpc_loss` podczas treningu potwierdza
zdolność modułu do nauki.

Kluczowe cechy:
- Używa JAX i Flax.
- Implementuje normalizację Z-score per-próbka, zgodnie z rekomendacją z analizy
  technicznej, aby ustabilizować trening.
- Wykorzystuje generator danych losowych dla pełnej izolacji i powtarzalności.
- Loguje stratę i normę gradientów w celu monitorowania zbieżności.

Jak uruchomić:
python pretrain_cpc.py --epochs 20 --learning_rate 1e-4
"""
import argparse
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
import numpy as np

# Import z projektu
from models.cpc import RealCPCEncoder, RealCPCConfig, temporal_info_nce_loss

# --- 1. KROK TRENINGOWY ---

@partial(jax.jit, static_argnums=(2,3))
def train_step(state: train_state.TrainState, batch: jnp.ndarray, k_prediction: int, temperature: float):
    """
    Pojedynczy krok treningowy dla enkodera CPC.
    """
    def loss_fn(params):
        # === KLUCZOWY KROK: NORMALIZACJA Z-SCORE PER-PRÓBKA ===
        # Zgodnie z rekomendacją, stabilizuje trening, zapobiegając dominacji
        # sygnałów o dużej amplitudzie w batchu.
        # Dla 3D input (batch, sequence, features) normalizujemy po wymiarze sequence
        mean = jnp.mean(batch, axis=1, keepdims=True)  # [batch, 1, features]
        std = jnp.std(batch, axis=1, keepdims=True) + 1e-8  # [batch, 1, features]
        normalized_batch = (batch - mean) / std

        # Obliczenie cech przez enkoder
        # RNG for dropout (deterministic across steps using folded key)
        step_val = getattr(state, 'step', jnp.array(0))
        dropout_rng = jax.random.fold_in(jax.random.PRNGKey(0), step_val)
        cpc_features = state.apply_fn(
            {'params': params},
            x=normalized_batch,
            training=True,
            rngs={'dropout': dropout_rng}
        )

        # Obliczenie straty
        loss = temporal_info_nce_loss(
            cpc_features,
            max_prediction_steps=k_prediction,
            temperature=temperature
        )
        return loss

    # Obliczenie straty i gradientów
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Aktualizacja stanu modelu
    new_state = state.apply_gradients(grads=grads)
    
    # Obliczenie normy gradientów do monitorowania
    grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
    
    return new_state, loss, grad_norm


# --- 2. GENERATOR DANYCH I GŁÓWNA PĘTLA ---

def dummy_data_generator(batch_size, sequence_length, key):
    """Generator losowych danych symulujących sygnały wejściowe."""
    while True:
        key, subkey = jax.random.split(key)
        # CPC oczekuje 3D input (batch, sequence, features)
        yield jax.random.normal(subkey, (batch_size, sequence_length, 1))


def main(args):
    """Główna funkcja uruchamiająca pre-trening."""
    print("--- Rozpoczynanie pre-treningu enkodera CPC ---")
    print(f"Konfiguracja: {args}")

    # Inicjalizacja kluczy PRNG
    key = jax.random.PRNGKey(args.seed)
    model_key, data_key, dropout_key = jax.random.split(key, 3)

    # Inicjalizacja modelu CPC
    cpc_config = RealCPCConfig(
        latent_dim=256,
        context_length=64,
        prediction_steps=args.k_prediction,
        temperature=args.temperature
    )
    encoder = RealCPCEncoder(config=cpc_config)
    
    # Inicjalizacja parametrów - CPC oczekuje 3D input (batch, sequence, features)
    dummy_input = jnp.ones((args.batch_size, args.sequence_length, 1))  # Dodaj wymiar features
    params = encoder.init({'params': model_key, 'dropout': dropout_key}, dummy_input, training=False)['params']

    # Inicjalizacja optymalizatora i stanu treningowego
    optimizer = optax.adamw(learning_rate=args.learning_rate)
    state = train_state.TrainState.create(
        apply_fn=encoder.apply,
        params=params,
        tx=optimizer
    )

    # Inicjalizacja generatora danych
    data_gen = dummy_data_generator(args.batch_size, args.sequence_length, data_key)

    # Główna pętla treningowa
    for epoch in range(args.epochs):
        epoch_losses = []
        epoch_grad_norms = []
        
        # Pętla po krokach w epoce
        step_iterator = tqdm(
            range(args.steps_per_epoch),
            desc=f"Epoka {epoch+1}/{args.epochs}",
            unit="krok"
        )
        for _ in step_iterator:
            batch = next(data_gen)
            state, loss, grad_norm = train_step(
                state,
                batch,
                k_prediction=args.k_prediction,
                temperature=args.temperature
            )
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)
        
        # Logowanie wyników epoki
        avg_loss = np.mean(epoch_losses)
        avg_grad_norm = np.mean(epoch_grad_norms)
        print(f"\n[Epoka {epoch+1}/{args.epochs}] "
              f"Średnia strata (cpc_loss): {avg_loss:.4f} | "
              f"Średnia norma gradientów: {avg_grad_norm:.4f}")

    print("\n--- Zakończono pre-trening ---")
    print("Obserwacja spadku wartości `cpc_loss` potwierdza, że model jest zdolny do nauki.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrypt do pre-treningu enkodera CPC.")
    
    # Argumenty dotyczące treningu
    parser.add_argument('--epochs', type=int, default=20, help='Liczba epok treningowych.')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Liczba kroków (batchy) na epokę.')
    parser.add_argument('--batch_size', type=int, default=64, help='Rozmiar batcha.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Współczynnik uczenia.')
    
    # Argumenty dotyczące danych i modelu
    parser.add_argument('--sequence_length', type=int, default=4096, help='Długość sekwencji wejściowej.')
    parser.add_argument('--k_prediction', type=int, default=4, help='Liczba kroków w przyszłość do przewidzenia.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperatura dla funkcji straty InfoNCE.')
    parser.add_argument('--seed', type=int, default=42, help='Ziarno losowości.')

    args = parser.parse_args()
    main(args)
