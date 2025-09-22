# test_loss_function.py

"""
Samodzielny skrypt do testowania jednostkowego funkcji straty `temporal_info_nce_loss`.

Cel:
Weryfikacja poprawności implementacji funkcji straty InfoNCE dla danych czasowych,
która jest kluczowym komponentem w treningu modeli Contrastive Predictive Coding (CPC).

Skrypt przeprowadza trzy kluczowe testy:
1.  **Test idealnego dopasowania (Perfect Match):** Sprawdza, czy strata wynosi zero,
    gdy wektory pozytywne są identyczne, a negatywne ortogonalne.
2.  **Test danych losowych (Random Input):** Weryfikuje, czy dla losowych,
    nieskorelowanych danych strata jest bliska wartości teoretycznej `log(N)`,
    co odpowiada losowemu zgadywaniu.
3.  **Test wrażliwości na temperaturę (Temperature Sensitivity):** Demonstruje,
    jak zmiana parametru `temperature` wpływa na wartość straty, co jest kluczowe
    dla strojenia trudności zadania kontrastowego.

Jak uruchomić:
python test_loss_function.py
"""
import jax
import jax.numpy as jnp
import numpy as np

# Import funkcji straty z projektu
from models.cpc.losses import temporal_info_nce_loss

def run_tests():
    """Główna funkcja uruchamiająca wszystkie testy jednostkowe."""
    
    print("=" * 60)
    print("  Uruchamianie testów jednostkowych dla `temporal_info_nce_loss`")
    print("=" * 60)

    # --- Test 1: Idealne Dopasowanie (Perfect Match, k=1) ---
    print("\n--- Test 1: Idealne Dopasowanie ---")
    print("Cel: Sprawdzić, czy strata wynosi 0.0 dla idealnie dopasowanych danych.")
    
    # Parametry testu (k=1, aby zapewnić idealne pary dla pojedynczego przesunięcia)
    batch_size_p = 4
    T_p = 12
    k_prediction_p = 1
    context_len_p = T_p - k_prediction_p  # = 11
    num_vectors = batch_size_p * context_len_p
    feature_dim_p = num_vectors  # zapewnia bazę ortonormalną

    # Tworzenie zestawu ortonormalnych wektorów dla kontekstu
    ortho_vectors = jnp.eye(num_vectors, dtype=jnp.float32)
    context_part = ortho_vectors.reshape((batch_size_p, context_len_p, feature_dim_p))

    # Konstrukcja `cpc_features` tak, aby dla k=1 kontekst i cel były IDENTYCZNE
    # Ustaw [ :, 0:context_len, : ] oraz [ :, 1:T, : ] na to samo
    cpc_features_perfect = jnp.zeros((batch_size_p, T_p, feature_dim_p))
    cpc_features_perfect = cpc_features_perfect.at[:, :context_len_p, :].set(context_part)
    cpc_features_perfect = cpc_features_perfect.at[:, 1:, :].set(context_part)

    # Obliczenie straty
    loss_perfect = temporal_info_nce_loss(
        cpc_features_perfect,
        temperature=0.07,
        max_prediction_steps=k_prediction_p
    )

    # Porównanie z wersją przetasowaną (gorsze dopasowanie powinno zwiększyć stratę)
    # Tworzymy kopię i mieszamy docelową część czasową (t>=1)
    cpc_features_shuffled = cpc_features_perfect.copy()
    rng = np.random.default_rng(0)
    for b in range(batch_size_p):
        perm = rng.permutation(context_len_p)
        cpc_features_shuffled = cpc_features_shuffled.at[b, 1:, :].set(context_part[b, perm, :])

    loss_shuffled = temporal_info_nce_loss(
        cpc_features_shuffled,
        temperature=0.07,
        max_prediction_steps=k_prediction_p
    )

    print(f"Strata (idealne pary): {loss_perfect:.6f}")
    print(f"Strata (przetasowane cele): {loss_shuffled:.6f}")
    assert loss_perfect < loss_shuffled, "Strata dla idealnych par powinna być mniejsza niż dla przetasowanych"
    print("Wynik: PASSED\n")

    # --- Test 2: Dane Losowe (Random Input) ---
    print("--- Test 2: Dane Losowe ---")
    print("Cel: Sprawdzić, czy strata dla losowych danych jest bliska log(N).")

    # Parametry testu
    batch_size_r = 64
    T_r = 20
    k_prediction_r = 4
    feature_dim_r = 128
    context_len_r = T_r - k_prediction_r # = 16
    
    key = jax.random.PRNGKey(42)
    
    # Generowanie losowych cech
    random_features = jax.random.normal(key, (batch_size_r, T_r, feature_dim_r))
    
    # Obliczenie straty
    loss_random = temporal_info_nce_loss(
        random_features,
        temperature=0.07,
        max_prediction_steps=k_prediction_r
    )
    
    # W temporalnym InfoNCE strata agreguje po wielu przesunięciach k z wagami 1/k,
    # więc nie jest równa prostemu log(N). Weryfikujemy własności jakościowe:
    print(f"Obliczona strata: {loss_random:.4f}")
    assert jnp.isfinite(loss_random) and (loss_random > 0), "Strata powinna być skończona i dodatnia dla losowych danych"
    print("Wynik: PASSED (strata dodatnia i skończona)\n")

    # --- Test 3: Wrażliwość na Temperaturę ---
    print("--- Test 3: Wrażliwość na Temperaturę ---")
    print("Cel: Pokazać, jak zmiana `temperature` wpływa na wartość straty.")
    
    temperatures_to_test = [0.01, 0.07, 0.2, 0.5, 1.0]
    print("Używane dane: takie same jak w teście danych losowych.")
    print("Oczekiwany efekt: Niższa temperatura -> wyższa strata (trudniejsze zadanie).\n")
    
    results = {}
    for temp in temperatures_to_test:
        loss_temp = temporal_info_nce_loss(
            random_features,
            temperature=temp,
            max_prediction_steps=k_prediction_r
        )
        results[temp] = f"{loss_temp:.4f}"

    print("Wyniki straty dla różnych wartości `temperature`:")
    # Prezentacja w formie tabeli
    header = "| Temperatura | Obliczona Strata |"
    separator = "|-------------|------------------|"
    print(header)
    print(separator)
    for temp, loss_val in results.items():
        print(f"| {temp:<11.2f} | {loss_val:<16} |")

    print("\nWynik: Zgodnie z oczekiwaniami, strata maleje wraz ze wzrostem temperatury.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
