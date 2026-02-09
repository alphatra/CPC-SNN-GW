# Status Projektu Inżynierskiego: CPC-SNN Gravitational Waves Detection

Dokument ten podsumowuje aktualny stan prac nad projektem, identyfikuje brakujące elementy oraz wyznacza kroki na najbliższą przyszłość.

## Update (2026-02-09): Decyzje po benchmarku OOD

- Zamrożony kandydat główny: `phase2_tf2d_tail_hn_v1` (TF2D + tail-aware + hard-negatives).
- Zamrożony kandydat OOD-robust: `ood_time_tf2d_tail_b` (najlepszy wynik strict train-early/test-late).
- `Primary KPI`: **TPR@FPR=1e-4**.
- `Secondary KPI`: TPR@1e-5 (do strojenia operacyjnego i kalibracji).
- Dla końcowej ścieżki TF2D:
  - przy FAR blisko `1e-4` preferowany jest wariant bez kalibracji,
  - przy FAR blisko `1e-5` preferowany jest wariant z kalibracją temperatury.
- Ścieżka 1D została wydzielona jako osobny task naprawczy i nie blokuje publikacji wyników TF2D.

## Update (2026-02-09): Results (Reports)

Finalna tabela benchmarków (A/B/C/C+calib):
- `reports/final_benchmark_table.md`

Pliki źródłowe użyte do wyników:
- `reports/bg_phase2_tf2d_base_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30_cal_temp.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30_nohn.txt`
- `reports/bg_phase2_tf2d_tail_hn_v1_swapped30_nohn_cal_temp.txt`
- `reports/calib_phase2_tf2d_tail_hn_v1_temp.json`
- `artifacts/final_candidate_ood_time_tf2d_tail_b/manifest.json`

Powtórzenia do estymacji `mean ± std`:
- `reports/repeats/a_base_r1.txt` ... `reports/repeats/a_base_r5.txt`
- `reports/repeats/b_tail_r1.txt` ... `reports/repeats/b_tail_r5.txt`
- `reports/repeats/c_tail_hn_r1.txt` ... `reports/repeats/c_tail_hn_r5.txt`
- `reports/repeats/c_tail_hn_caltemp_r1.txt` ... `reports/repeats/c_tail_hn_caltemp_r5.txt`

Skrypt reprodukcji i formalnego "freeze":
- `./scripts/reproduce_phase2_abccalib.sh`
- wariant pełny (z retrainingiem A/B/C): `./scripts/reproduce_phase2_abccalib.sh --with-train`

Skrypt twardego testu OOD (train-early / test-late):
- `./scripts/run_ood_time_protocol.sh`
- tryb run-based OOD: `./scripts/run_ood_time_protocol.sh --mode run --run-map <id_to_run.json> --train-runs <runs> --test-runs <runs>`

## 1. Co zostało zrobione (Completed)

### A. Pipeline Danych (Data Handling)
Zakończono gruntowną refaktoryzację modułu generowania i obsługi danych.
- **Generowanie Danych (`generate_training_set.py`)**:
    - Zaimplementowano obsługę **Multi-IFO** (wiele detektorów: H1, L1).
    - Wdrożono system konfiguracji oparty na **Hydra** (`configs/data_generation.yaml`).
    - Naprawiono błędy numeryczne (PSD z `scipy.signal.welch`, rzutowanie typów `LIGOTimeGPS`).
    - Dodano projekcję sygnałów (Waveform Projection) uwzględniającą opóźnienia geometryczne i charakterystykę anten.
    - **Nowość**: Zaimplementowano **Glitch Injection** (wstrzykiwanie artefaktów typu Sine-Gaussian) do próbek tła, aby zwiększyć odporność modelu.
    - **Optymalizacja**: Dodano mechanizm `fetch_with_retry` (ponawianie prób pobierania) w celu obsługi niestabilności serwerów GWOSC (`ReadTimeout`).
    - **Optymalizacja**: Skrypt teraz **wznawia generowanie** (tryb `append`) zamiast nadpisywać plik, co pozwala na dokończenie przerwanego procesu.
    - **Weryfikacja**: Wygenerowano poprawny "mini dataset" (200 próbek) oraz częściowy zbiór treningowy (ok. 5800 próbek), który zawiera poprawne metadane i strukturę.
- **Preprocessing Sygnałów**:
    - Zestaw funkcji pomocniczych do przetwarzania sygnałów (STFT, Whitening) jest gotowy i przetestowany.

### B. Wizualizacja i Diagnostyka
- **Notebooki**:
    - `notebooks/03_Data_Pipeline_Visualization.ipynb`: Wizualna inspekcja danych i preprocessingu.
    - **Nowość**: `notebooks/04_Spike_Encoding_Analysis.ipynb`: Analiza i animacja procesu konwersji fal grawitacyjnych na spiki (Delta Modulation) z kontekstem fizycznym.
- **Struktura Projektu**:

    - Uporządkowano katalogi (`src/`, `configs/`, `data/`, `notebooks/`).
    - Usunięto zbędne skrypty tymczasowe.

### C. Modele i Architektura (Models)
- **Modularny System CPC-SNN (`src/models/`)**:
   ### Faza 2: Architektura SNN (Zakończona)
- [x] Implementacja `DeltaModulationEncoder`.
- [x] Implementacja `SpikingCNN` (Feature Extractor).
- [x] Implementacja `RSNN` (Context Network) - **Zrobione** (zastąpiono GRU).
- [x] Integracja w `CPCSNN`.
- [x] Weryfikacja (Smoke Test) - **Zaliczone**.
- [x] Aktualizacja Notebooków - **Zrobione** (naprawiono ścieżki i normalizację).

### Faza 3: Detekcja Anomalii (W toku)
- [x] Przygotowanie danych (podział na szum/sygnał).
- [x] Trening modelu na szumie (`src/train/train_cpc.py`) - **Zweryfikowane** (Fast Run OK, W&B, Checkpoints).
- [x] Implementacja skryptu inferencji (`src/inference_anomaly.py`) - **Zrobione**.
- [x] Notebook demonstracyjny (`notebooks/05_Anomaly_Detection_Inference.ipynb`) - **Zrobione**.
- [x] Weryfikacja detekcji (czy loss skacze na sygnale?) - **Wstępnie zweryfikowane** (skrypt ewaluacji działa).

### Faza 4: Ewaluacja i Eksperymenty (W toku)
- [x] Skrypty ewaluacyjne istnieją i działają jako entrypointy:
  - `src/evaluation/evaluate_snn.py` (ewaluacja klasyfikacji binarnej)
  - `src/evaluation/evaluate_background.py` (ewaluacja tła / tails / time-slides)
- [x] Dodano kompatybilność ładowania starszych checkpointów (`src/evaluation/model_loader.py`).
- [ ] Potrzebny pełny benchmark low-FPR (z większym zbiorem tła) i finalne porównanie wariantów modelu.

## 2. Czego brakuje (Missing / To Do)

### A. Pełny Zbiór Danych
- **Status**: Zakończono generowanie. Posiadamy `data/cpc_snn_train.h5` z ~10k+ próbek.
- **Weryfikacja**: Indeksy `data/indices_noise.json` i `data/indices_signal.json` są spójne z HDF5.

### B. Trening Modelu (End-to-End)
- **Status**: Główny tor treningowy to `python -m src.train.train_cpc`.
- **Uwaga**: Stary tor MVP (`src/train/train_mvp.py`, `src/train/eval_mvp.py`) został oznaczony jako **deprecated** i nie jest już ścieżką produkcyjną.

### C. Ewaluacja i Metryki
- Skrypty ewaluacyjne są dostępne, ale wymagają finalnej walidacji wyników:
  - ROC-AUC / PR-AUC,
  - TPR@FPR (1e-3, 1e-4, 1e-5, 1e-6),
  - kalibracja (ECE, Brier),
  - tails i stabilność tła.

### D. Eksperymenty SNN
- Nadal do domknięcia: systematyczne porównanie wariantów (`use_tf2d`, `use_sft`, 1D raw/spikes) oraz finalny raport eksperymentów.

## 3. Następne Kroki (Next Steps)

### Krok 1: Stabilny benchmark
Uruchomić jednolity protokół: `train_cpc` -> `evaluate_snn`/`evaluate_background` na tych samych splitach.

### Krok 2: Domknięcie metryk low-FPR
Zwiększyć pulę tła w ewaluacji, aby FPR=1e-4 i niższe były mierzalne bez "Below Resolution".

### Krok 3: Tuning
Eksperymenty z:
- architekturą (hidden/context dim, head),
- parametrami CPC (prediction steps, temperature, lambda_infonce),
- augmentacjami i strategią splitu (`time` vs `random`).

## 4. Podsumowanie dla Promotora
Projekt posiada działający pipeline danych i treningu oraz zestaw checkpointów eksperymentalnych. Aktualny etap to stabilizacja ścieżki eksperymentalnej i domknięcie jakości ewaluacji na niskich FPR.
