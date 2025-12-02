# Status Projektu Inżynierskiego: CPC-SNN Gravitational Waves Detection

Dokument ten podsumowuje aktualny stan prac nad projektem, identyfikuje brakujące elementy oraz wyznacza kroki na najbliższą przyszłość.

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
- [x] Skrypt ewaluacji (`src/evaluation/evaluate_anomaly.py`) - **Zrobione** (liczy ROC/AUC).
- [ ] Pełny trening modelu (na GPU/MPS) - **W toku** (User uruchomił trening).
- [ ] Analiza wyników (ROC, AUC) na pełnym zbiorze.
- [x] Analiza koincydencji (H1 + L1) - **Skrypty gotowe** (`src/evaluation/coincidence_analysis.py`). Wymaga wytrenowania osobnych modeli dla H1 i L1.ugą funkcji straty InfoNCE.

## 2. Czego brakuje (Missing / To Do)

### A. Pełny Zbiór Danych
- **Status**: Zakończono generowanie. Posiadamy `cpc_snn_train.h5` z **10,000 próbek**.
- **Weryfikacja**: Potwierdzono liczbę próbek w pliku.

### B. Trening Modelu (End-to-End)
- **Status**: Przeprowadzono pomyślny "Smoke Test" (`train_smoke_test.py`).
- Pipeline działa: Ładowanie danych (z rekonstrukcją ISTFT) -> Model CPC-SNN -> Loss -> Backward pass.

### C. Ewaluacja i Metryki
- Brakuje dedykowanego skryptu lub modułu do **ewaluacji modelu**.
- Potrzebne metryki:
    - Accuracy (dla klasyfikacji binarnej SNN).
    - ROC-AUC (krzywa charakterystyki operacyjnej).
    - Latency (czas reakcji modelu - kluczowe dla SNN).
    - Confusion Matrix.

### D. Eksperymenty SNN
- Należy zweryfikować poprawność działania warstwy SNN (Spiking Neural Network). Czy neurony faktycznie "strzelają"? Czy kodowanie CPC działa zgodnie z założeniami?

## 3. Następne Kroki (Next Steps)

### Krok 1: Pełny Trening
Uruchomienie długiego treningu na pełnym zbiorze 10k próbek.
- Należy zoptymalizować pętlę treningową (np. użycie GPU, `torch.compile`, lub skrócenie sekwencji czasowych).

### Krok 2: Implementacja Ewaluacji
Stworzenie skryptu `evaluate.py` lub notebooka, który:

Stworzenie skryptu `evaluate.py` lub notebooka, który:
1. Ładuje wytrenowany model.
2. Przechodzi przez zbiór testowy.
3. Rysuje krzywą ROC i liczy AUC.

### Krok 4: Tuning Hiperparametrów
Eksperymenty z:
- Architekturą sieci (liczba warstw, neuronów).
- Parametrami CPC (kroki predykcji, temperatura).
- Parametrami danych (długość okna STFT, zakres częstotliwości).

## 4. Podsumowanie dla Promotora
Projekt posiada solidne fundamenty w postaci działającego i nowoczesnego pipeline'u danych (Multi-IFO, Hydra). Główny nacisk należy teraz przenieść z inżynierii danych na inżynierię modelu (trening, weryfikacja SNN, wyniki).
