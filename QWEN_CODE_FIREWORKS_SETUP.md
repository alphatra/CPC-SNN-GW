# Qwen Code CLI + Fireworks.ai - Kompletna Konfiguracja

## Przegląd
Ten dokument opisuje jak skonfigurować Qwen Code CLI do pracy z Fireworks.ai jako providerem modeli AI dla zadań kodowania i analizy.

## Wymagania
- Node.js v20+ 
- Konto Fireworks.ai z API key
- Dostęp do modeli Kimi K2 lub Qwen3 na Fireworks.ai

## Instalacja

### 1. Sprawdź Node.js
```bash
node --version
# Wymagane: v20.0.0 lub wyżej
```

### 2. Zainstaluj Qwen Code CLI
```bash
npm install -g @qwen-code/qwen-code
xd
```

## Konfiguracja Fireworks.ai

### Dostępne Modele
- **Kimi K2 Instruct**: `accounts/fireworks/models/kimi-k2-instruct`
- **Qwen3-30B**: `accounts/fireworks/models/qwen3-30b-a3b`

### Zmienne Środowiskowe

#### Opcja 1: Ustawienie w sesji bash
```bash
export OPENAI_API_KEY="fw_3ZWp39RpKqkTuVNsmcQs2U8m"
export OPENAI_BASE_URL="https://api.fireworks.ai/inference/v1"
export OPENAI_MODEL="accounts/fireworks/models/kimi-k2-instruct"
```

#### Opcja 2: Trwała konfiguracja w ~/.bashrc lub ~/.zshrc
```bash
echo 'export OPENAI_API_KEY="fw_3ZWp39RpKqkTuVNsmcQs2U8m"' >> ~/.bashrc
echo 'export OPENAI_BASE_URL="https://api.fireworks.ai/inference/v1"' >> ~/.bashrc
echo 'export OPENAI_MODEL="accounts/fireworks/models/kimi-k2-instruct"' >> ~/.bashrc
source ~/.bashrc
```

#### Opcja 3: Lokalna konfiguracja projektu (.env)
Utwórz plik `.env` w katalogu projektu:
```bash
OPENAI_API_KEY="fw_3ZWp39RpKqkTuVNsmcQs2U8m"
OPENAI_BASE_URL="https://api.fireworks.ai/inference/v1"
OPENAI_MODEL="accounts/fireworks/models/kimi-k2-instruct"
```

## Weryfikacja Konfiguracji

### Test Podstawowy
```bash
echo "Describe the main components of this project" | qwen
```

### Test Nieinteraktywny
```bash
qwen -p "Przeanalizuj strukturę tego projektu i wyjaśnij główne komponenty"
```

### Sprawdzenie Zmiennych
```bash
echo "API Key: $OPENAI_API_KEY"
echo "Base URL: $OPENAI_BASE_URL"
echo "Model: $OPENAI_MODEL"
```

## Główne Funkcje Qwen Code CLI

### Tryby Użycia

1. **Interaktywny**:
   ```bash
   qwen
   ```

2. **Nieinteraktywny**:
   ```bash
   qwen -p "What are the security implications of this code?"
   ```

3. **Prompt + Kontynuacja Interaktywna**:
   ```bash
   qwen -i "Analyze this file structure"
   ```

### Przydatne Opcje

- `--all-files`: Uwzględnia wszystkie pliki w kontekście
- `--debug`: Tryb debugowania
- `--yolo`: Automatyczne akceptowanie akcji
- `--openai-logging`: Logowanie wywołań API
- `--show-memory-usage`: Pokazuje użycie pamięci

### Przykłady Użycia

#### Analiza Kodu
```bash
qwen -p "Zidentyfikuj potencjalne problemy bezpieczeństwa w tym kodzie"
```

#### Refaktoryzacja
```bash
qwen -p "Zaproponuj refaktoryzację tej funkcji dla lepszej czytelności"
```

#### Dokumentacja
```bash
qwen -p "Wygeneruj dokumentację JSDoc dla tego modułu"
```

#### Debugowanie
```bash
qwen -p "Pomóż mi znaleźć przyczynę tego błędu: [opis błędu]"
```

## Rozwiązywanie Problemów

### Problem: "Model not found"
- Sprawdź czy model jest dostępny na Fireworks.ai
- Zweryfikuj poprawność nazwy modelu
- Sprawdź czy masz dostęp do modelu

### Problem: "Authentication failed"
- Zweryfikuj API key
- Sprawdź czy key nie wygasł
- Sprawdź czy masz odpowiednie uprawnienia

### Problem: "Connection timeout"
- Sprawdź połączenie internetowe
- Zweryfikuj URL API
- Sprawdź status Fireworks.ai

## Porównanie Modeli

### Kimi K2 Instruct
- **Zalety**: Bardzo dobra analiza kodu, szybkie odpowiedzi
- **Użycie**: Ogólne zadania kodowania, refaktoryzacja
- **Model ID**: `accounts/fireworks/models/kimi-k2-instruct`

### Qwen3-30B 
- **Zalety**: Większy model, więcej parametrów
- **Użycie**: Złożone analizy, duże projekty
- **Model ID**: `accounts/fireworks/models/qwen3-30b-a3b`

## Status Konfiguracji
✅ **SUKCES**: Qwen Code CLI v0.0.1-alpha.10 skonfigurowany z Fireworks.ai  
✅ **ZWERYFIKOWANY**: Kimi K2 Instruct działa poprawnie  
✅ **GOTOWY**: Do użycia w projektach kodowania

## Dodatkowe Zasoby
- [Qwen Code GitHub](https://github.com/QwenLM/qwen-code)
- [Fireworks.ai Documentation](https://docs.fireworks.ai/)
- [API Reference](https://docs.fireworks.ai/api-reference)

---
*Dokument utworzony: 2025-01-28*  
*Wersja Qwen Code: 0.0.1-alpha.10*  
*Provider: Fireworks.ai* 