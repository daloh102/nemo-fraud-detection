import os
import json
import pytest
from unittest.mock import patch, MagicMock
from src.curator.data_generator import generate_and_contaminate

# ==============================================================================
# 1. FIXTURES & MOCKS (Verhindert echte GPU / Container Requests)
# ==============================================================================

@pytest.fixture
def mock_nemo_sdg():
    """Mockt das NeMo SDK / LLM-Client, damit keine echten GPU-Requests gesendet werden."""
    with patch("src.curator.data_generator.LLMClient") as mock_client, \
         patch("src.curator.data_generator.SyntheticDataGenerator") as mock_generator, \
         patch("src.curator.data_generator.DocumentDataset") as mock_dataset:
        
        # Erzeuge synthetische Mock-Antworten für 10 Test-Items
        mock_gen_instance = MagicMock()
        mock_generator.return_value = mock_gen_instance
        
        mock_res_dataset = MagicMock()
        mock_res_dataset.to_dict.return_value = [
            {
                "id": f"CALL_{i:05d}",
                "response": json.dumps({
                    "transcript": [
                        {"speaker": "Agent", "text": "Guten Tag, Kundenservice."},
                        {"speaker": "Customer", "text": "Hallo, ich habe eine Frage zu meinem Konto."}
                    ]
                })
            }
            for i in range(10)
        ]
        mock_gen_instance.generate.return_value = mock_res_dataset
        yield mock_generator


@pytest.fixture
def generated_test_files(tmp_path, mock_nemo_sdg):
    """
    Führt generate_and_contaminate() einmalig mit Mock-Daten aus 
    und stellt die temporären Dateipfade für die Tests bereit.
    """
    test_raw_path = tmp_path / "raw" / "call_transcripts_raw.jsonl"
    test_bench_path = tmp_path / "evaluation" / "benchmark_dataset.jsonl"
    
    # Verzeichnisse anlegen
    test_raw_path.parent.mkdir(parents=True, exist_ok=True)
    test_bench_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pfade in data_generator patchen
    with patch("src.curator.data_generator.RAW_OUT_PATH", str(test_raw_path)), \
         patch("src.curator.data_generator.BENCHMARK_PATH", str(test_bench_path)), \
         patch("src.curator.data_generator.TOTAL_RECORDS", 10):
        
        generate_and_contaminate()
    
    return {
        "raw": test_raw_path,
        "benchmark": test_bench_path
    }


# ==============================================================================
# 2. FORMAT & SCHEMA TESTS (Quality Gate 2)
# ==============================================================================

def test_files_exist(generated_test_files):
    """Prüft, ob die Ausgabedateien erfolgreich angelegt wurden."""
    raw_path = generated_test_files["raw"]
    bench_path = generated_test_files["benchmark"]
    
    assert raw_path.exists(), "Rohdaten-Datei wurde nicht erstellt!"
    assert bench_path.exists(), "Benchmark-Datei wurde nicht erstellt!"


def test_jsonl_validity_and_schema(generated_test_files):
    """Prüft, ob jede Zeile in der Rohdaten-Datei valides JSON ist und die Pflichtfelder enthält."""
    raw_path = generated_test_files["raw"]
    
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    assert len(lines) == 10, f"Erwartet wurden 10 Zeilen, erhalten: {len(lines)}"

    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            pytest.fail(f"Zeile {i} ist kein valides JSON!")
        
        # Schema-Prüfung der Felder
        assert "call_id" in data, f"Zeile {i} hat keine 'call_id'"
        assert "text" in data, f"Zeile {i} hat kein 'text'-Feld"
        assert isinstance(data["text"], str), f"Zeile {i}: 'text' ist kein String"


def test_benchmark_alignment(generated_test_files):
    """Prüft, ob das Benchmark-Dataset korrekte Ground-Truth Labels enthält."""
    bench_path = generated_test_files["benchmark"]
    
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_lines = [json.loads(l) for l in f]
    
    assert len(bench_lines) == 10, "Benchmark-Datei hat nicht die erwartete Anzahl an Einträgen!"

    valid_labels = {"legit", "fraud"}
    for i, data in enumerate(bench_lines):
        assert "ground_truth" in data, f"Zeile {i} im Benchmark fehlt 'ground_truth'"
        assert data["ground_truth"] in valid_labels, f"Ungültiges Label in Zeile {i}: {data['ground_truth']}"