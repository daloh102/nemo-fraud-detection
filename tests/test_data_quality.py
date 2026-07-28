import json
import os
import pytest

RAW_PATH = "data/raw/call_transcripts_raw.jsonl"
BENCHMARK_PATH = "data/evaluation/benchmark_dataset.jsonl"


# ==============================================================================
# 1. FIXTURES (Laden der echten Daten)
# ==============================================================================

@pytest.fixture(scope="module")
def raw_data():
    """Lädt die erzeugten Rohdaten für alle Tests in diesem Modul."""
    assert os.path.exists(RAW_PATH), (
        f"Rohdaten-Datei {RAW_PATH} nicht gefunden! "
        f"Hast du das Data-Generator-Skript für die 10.000 Gespräche schon ausgeführt?"
    )
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records


@pytest.fixture(scope="module")
def benchmark_data():
    """Lädt das erzeugte Benchmark-Dataset."""
    assert os.path.exists(BENCHMARK_PATH), f"Benchmark-Datei {BENCHMARK_PATH} nicht gefunden!"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records


# ==============================================================================
# 2. QUALITY GATE 3 TESTS
# ==============================================================================

def test_total_record_count(raw_data, benchmark_data):
    """Prüft, ob die erwartete Datenmenge erzeugt wurde und Datenbestände synchron sind."""
    assert len(raw_data) > 0, "Rohdatensatz ist vollkommen leer!"
    # Falls du z.B. exakt 10.000 Datensätze erwarten würdest:
    # assert len(raw_data) >= 10000, f"Zu wenige Datensätze: {len(raw_data)}"
    
    assert len(raw_data) == len(benchmark_data), (
        f"Inkonsistenz: Rohdaten ({len(raw_data)}) und Benchmark ({len(benchmark_data)}) "
        f"haben unterschiedliche Zeilenzahlen!"
    )


def test_text_length_distribution(raw_data):
    """Stellt sicher, dass die Dialoge nicht unvollständig oder leer sind."""
    empty_or_short_texts = [r for r in raw_data if len(r.get("text", "").strip()) < 20]
    
    # Maximal 1% extrem kurze/leere Texte als Rauschen tolerieren
    max_allowed_short = len(raw_data) * 0.01
    assert len(empty_or_short_texts) <= max_allowed_short, (
        f"Zu viele leere oder extrem kurze Transkripte gefunden: {len(empty_or_short_texts)}"
    )


def test_noise_injection_presence(raw_data):
    """
    Prüft, ob die gezielte Kontaminierung (Duplikate, PII, Müll) 
    tatsächlich in den Daten vorkommt.
    """
    # 1. Duplikate (_DUP Suffix in call_id)
    has_duplicates = any("_DUP" in r.get("call_id", "") for r in raw_data)
    
    # 2. PII-Injektionen (Sensible Daten wie IBAN oder Kreditkartennummern)
    has_pii = any(
        "DE" in r.get("text", "") or "Kreditkarte" in r.get("text", "") or "IBAN" in r.get("text", "")
        for r in raw_data
    )

    # 3. Garbage / Unstrukturierter Noise (z.B. Tippfehler/Müll)
    has_garbage = any(len(r.get("text", "")) < 15 for r in raw_data)

    assert has_duplicates, "Keine Duplikate im Datensatz gefunden! Kontaminierung prüfen."
    assert has_pii, "Keine PII-Daten (z.B. IBAN/Kreditkarte) im Datensatz gefunden!"
    assert has_garbage, "Kein Datenmüll/Garbage im Datensatz gefunden!"


def test_benchmark_label_distribution(benchmark_data):
    """Prüft die Verteilung von Legit vs. Fraud Labels im Benchmark-Set."""
    labels = [r.get("ground_truth") for r in benchmark_data]
    
    legit_count = labels.count("legit")
    fraud_count = labels.count("fraud")

    assert legit_count > 0, "Keine 'legit'-Labels im Benchmark-Datensatz!"
    assert fraud_count > 0, "Keine 'fraud'-Labels im Benchmark-Datensatz!"

    # Beispiel für eine Verhältnismäßigkeitsprüfung (z.B. keine Seite völlig unberücksichtigt)
    ratio = fraud_count / len(benchmark_data)
    assert 0.05 <= ratio <= 0.95, f"Extrem unbalanciertes Label-Verhältnis: Fraud-Anteil liegt bei {ratio:.2%}"