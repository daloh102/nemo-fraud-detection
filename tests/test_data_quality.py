import json
import os
from pathlib import Path
import pytest

# --------------------------------------------------------------------------
# 1. PFAD- & ORDNER-KONFIGURATION
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent if "tests" in str(Path(__file__)) else Path("/data/nemo-fraud-detection")
DATA_DIR = BASE_DIR / "data"

NOISY_RAW_PATH = DATA_DIR / "raw" / "dialogues_transcripts_noisy.jsonl"
BENCHMARK_PATH = DATA_DIR / "evaluation" / "dialogues_benchmark.jsonl"


# ==============================================================================
# 1. FIXTURES (Laden der echten Daten)
# ==============================================================================

@pytest.fixture(scope="module")
def raw_data():
    """Lädt die kontaminierten Rohdaten (dialogues_transcripts_noisy.jsonl)."""
    assert NOISY_RAW_PATH.exists(), (
        f"Rohdaten-Datei {NOISY_RAW_PATH} nicht gefunden! "
        f"Hast du das Injektions-Skript (inject_curator_noise.py) schon ausgeführt?"
    )
    records = []
    with open(NOISY_RAW_PATH, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    pytest.fail(f"Zeile {line_no} in {NOISY_RAW_PATH.name} ist kein valides JSON: {e}")
    return records


@pytest.fixture(scope="module")
def benchmark_data():
    """Lädt das Referenz-Benchmark-Dataset (dialogues_benchmark.jsonl)."""
    assert BENCHMARK_PATH.exists(), f"Benchmark-Datei {BENCHMARK_PATH} nicht gefunden!"
    records = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ==============================================================================
# 2. QUALITY GATE 3 TESTS
# ==============================================================================

def test_nemo_curator_schema_compatibility(raw_data):
    """
    Stellt sicher, dass die Noisy-Daten das von NeMo Curator erwartete
    Eingabeformat besitzen (z. B. Vorhandensein des Pflichtfeldes 'text').
    """
    assert len(raw_data) > 0, "Der Noisy-Datensatz ist vollkommen leer!"
    
    missing_text_key = [r for r in raw_data if "text" not in r]
    assert len(missing_text_key) == 0, (
        f"Gefunden: {len(missing_text_key)} Einträge ohne den Pflicht-Key 'text' für NeMo Curator!"
    )


def test_record_count_and_relations(raw_data, benchmark_data):
    """
    Prüft die Datenmenge. Da Duplikate injiziert wurden, muss der 
    Noisy-Datensatz mindestens so viele Zeilen wie das Benchmark-Set haben.
    """
    assert len(benchmark_data) > 0, "Benchmark-Datensatz ist leer!"
    
    # Durch Injektion von Duplikaten muss len(raw_data) >= len(benchmark_data) sein
    assert len(raw_data) >= len(benchmark_data), (
        f"Inkonsistenz: Noisy-Daten ({len(raw_data)}) haben weniger Zeilen "
        f"als das Benchmark-Set ({len(benchmark_data)})!"
    )


def test_text_length_distribution(raw_data):
    """Stellt sicher, dass sowohl verwertbarer Text als auch injizierter Müll vorhanden sind."""
    empty_or_short_texts = [r for r in raw_data if len(r.get("text", "").strip()) < 20]
    
    # Müll-Texte wurden bewusst injiziert, sollten aber maximal 5% der Gesamtmenge ausmachen
    max_allowed_short = len(raw_data) * 0.05
    assert len(empty_or_short_texts) <= max_allowed_short, (
        f"Zu hohe Quote an extrem kurzen/leeren Texten: {len(empty_or_short_texts)} (Max erlaubt: {max_allowed_short})"
    )


def test_noise_injection_presence(raw_data):
    """
    Prüft, ob alle Typen gezielter Kontaminierung (Duplikate, PII, Müll, Encoding-Fehler)
    tatsächlich im Datensatz für den Curator-Test vorhanden sind.
    """
    # 1. Duplikate (_DUP Suffix in call_id)
    has_duplicates = any("_DUP" in str(r.get("call_id", "")) for r in raw_data)
    
    # 2. PII-Injektionen (IBAN, Kreditkarten, E-Mails etc.)
    has_pii = any(
        any(keyword in r.get("text", "") for keyword in ["Kreditkarte", "IBAN", "DE89", "@example.de"])
        for r in raw_data
    )

    # 3. Garbage / Unstrukturierter Noise
    has_garbage = any(r.get("text", "").strip() in ["???", "N/A", "NULL", "a", "asdfghjkl;"] for r in raw_data)

    # 4. Encoding-Fehler (Mojibake)
    has_encoding_errors = any("Ã¤" in r.get("text", "") or "Ã¶" in r.get("text", "") for r in raw_data)

    assert has_duplicates, "Keine Duplikate im Noisy-Datensatz gefunden!"
    assert has_pii, "Keine PII-Daten (z. B. IBAN/Kreditkarte) im Noisy-Datensatz gefunden!"
    assert has_garbage, "Kein Datenmüll/Garbage im Noisy-Datensatz gefunden!"
    assert has_encoding_errors, "Keine UTF-8 Encoding-Fehler (Mojibake) im Noisy-Datensatz gefunden!"


def test_benchmark_label_distribution(benchmark_data):
    """Prüft die Verteilung von Legit vs. Fraud Labels im Unberührten Benchmark-Set."""
    labels = [r.get("ground_truth", r.get("label")) for r in benchmark_data]
    
    legit_count = labels.count("legitimate")
    fraud_count = labels.count("fraud")

    assert legit_count > 0, "Keine 'legit'-Labels im Benchmark-Datensatz!"
    assert fraud_count > 0, "Keine 'fraud'-Labels im Benchmark-Datensatz!"

    ratio = fraud_count / len(benchmark_data)
    assert 0.05 <= ratio <= 0.95, f"Extrem unbalanciertes Label-Verhältnis im Benchmark: Fraud-Anteil liegt bei {ratio:.2%}"