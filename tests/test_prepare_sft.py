import json
import pytest
from unittest.mock import patch
from src.curator.prepare_sft import main


# ==============================================================================
# 1. HAPPY PATH TEST
# ==============================================================================
def test_prepare_sft_pipeline(tmp_path):
    """
    Testet den regulären Ablauf der SFT-Datensatz-Vorbereitung mit voll
    übereinstimmenden IDs zwischen Curated- und Benchmark-Dateien.
    """
    curated_input = tmp_path / "curated.jsonl"
    benchmark_input = tmp_path / "benchmark.jsonl"
    sft_dir = tmp_path / "sft"

    curated_data = []
    benchmark_data = []

    for i in range(10):
        cid = f"doc-{i:05d}"
        label = "fraud" if i % 2 == 0 else "legit"

        # Key-Mapping: call_id im Curated-File entspricht id im Benchmark-File
        curated_data.append({
            "call_id": cid,
            "id": f"doc_id-{i:05d}",
            "text": f"Das ist ein Test-Transkript für Call {i}."
        })
        benchmark_data.append({
            "id": cid,
            "label": label
        })

    with open(curated_input, "w", encoding="utf-8") as f:
        for d in curated_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(benchmark_input, "w", encoding="utf-8") as f:
        for d in benchmark_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with patch("src.curator.prepare_sft.CURATED_PATH", curated_input), \
         patch("src.curator.prepare_sft.BENCHMARK_PATH", benchmark_input), \
         patch("src.curator.prepare_sft.SFT_DIR", sft_dir):

        main()

    train_file = sft_dir / "train.jsonl"
    val_file = sft_dir / "validation.jsonl"
    test_file = sft_dir / "test.jsonl"

    assert train_file.exists()
    assert val_file.exists()
    assert test_file.exists()

    def count_lines(p):
        with open(p, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    total_exported = count_lines(train_file) + count_lines(val_file) + count_lines(test_file)
    assert total_exported == 10

    with open(train_file, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "input" in first_line
        assert "output" in first_line
        assert first_line["output"] in ["fraud", "legit"]


# ==============================================================================
# 2. FAIL-FAST TEST: FEHLENDE BENCHMARK DATEI
# ==============================================================================
def test_prepare_sft_missing_benchmark_raises_error(tmp_path):
    """
    Stellt sicher, dass das Skript sofort mit SystemExit(1) abbricht,
    falls die Benchmark-Datei nicht existiert.
    """
    curated_input = tmp_path / "curated.jsonl"
    missing_benchmark = tmp_path / "does_not_exist.jsonl"
    sft_dir = tmp_path / "sft"

    curated_data = [{"call_id": "doc-00001", "text": "Test"}]
    with open(curated_input, "w", encoding="utf-8") as f:
        for d in curated_data:
            f.write(json.dumps(d) + "\n")

    with patch("src.curator.prepare_sft.CURATED_PATH", curated_input), \
         patch("src.curator.prepare_sft.BENCHMARK_PATH", missing_benchmark), \
         patch("src.curator.prepare_sft.SFT_DIR", sft_dir):

        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1


# ==============================================================================
# 3. DATA LEAKAGE PRÜFUNG (TRAIN / VAL / TEST OVERLAP)
# ==============================================================================
def test_prepare_sft_no_data_leakage(tmp_path):
    """
    Stellt sicher, dass kein Datensatz gleichzeitig in Train, Val oder Test landet.
    """
    curated_input = tmp_path / "curated.jsonl"
    benchmark_input = tmp_path / "benchmark.jsonl"
    sft_dir = tmp_path / "sft"

    curated_data = []
    benchmark_data = []

    for i in range(20):
        cid = f"doc-{i:05d}"
        curated_data.append({"call_id": cid, "text": f"Eindeutiger Text {i}"})
        benchmark_data.append({"id": cid, "label": "legit"})

    with open(curated_input, "w", encoding="utf-8") as f:
        for d in curated_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(benchmark_input, "w", encoding="utf-8") as f:
        for d in benchmark_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with patch("src.curator.prepare_sft.CURATED_PATH", curated_input), \
         patch("src.curator.prepare_sft.BENCHMARK_PATH", benchmark_input), \
         patch("src.curator.prepare_sft.SFT_DIR", sft_dir):

        main()

        def get_inputs(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return {json.loads(line)["input"] for line in f if line.strip()}

        train_inputs = get_inputs(sft_dir / "train.jsonl")
        test_inputs = get_inputs(sft_dir / "test.jsonl")
        val_inputs = get_inputs(sft_dir / "validation.jsonl")

        assert len(train_inputs.intersection(test_inputs)) == 0
        assert len(train_inputs.intersection(val_inputs)) == 0
        assert len(val_inputs.intersection(test_inputs)) == 0


# ==============================================================================
# 4. ROBUSTHEIT GEGENÜBER LEERZEILEN
# ==============================================================================
def test_prepare_sft_handles_empty_lines(tmp_path):
    """
    Testet, ob leere Zeilen in den JSONL-Dateien korrekt ignoriert werden.
    """
    curated_input = tmp_path / "curated.jsonl"
    benchmark_input = tmp_path / "benchmark.jsonl"
    sft_dir = tmp_path / "sft"

    with open(curated_input, "w", encoding="utf-8") as f:
        f.write('{"call_id": "doc-01", "text": "Hallo"}\n\n\n{"call_id": "doc-02", "text": "Welt"}\n')

    with open(benchmark_input, "w", encoding="utf-8") as f:
        f.write('{"id": "doc-01", "label": "legit"}\n\n{"id": "doc-02", "label": "fraud"}\n')

    with patch("src.curator.prepare_sft.CURATED_PATH", curated_input), \
         patch("src.curator.prepare_sft.BENCHMARK_PATH", benchmark_input), \
         patch("src.curator.prepare_sft.SFT_DIR", sft_dir):

        main()

        def count_lines(p):
            with open(p, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())

        total = count_lines(sft_dir / "train.jsonl") + \
                count_lines(sft_dir / "validation.jsonl") + \
                count_lines(sft_dir / "test.jsonl")

        assert total == 2