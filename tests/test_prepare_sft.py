import os
import json
import pytest
from unittest.mock import patch
from src.customizer.prepare_sft import prepare_sft_splits

def test_prepare_sft_splits(tmp_path):
    curated_input = tmp_path / "deduplicated_calls.jsonl"
    benchmark_input = tmp_path / "benchmark_dataset.jsonl"
    sft_dir = tmp_path / "sft"
    
    # Mindestens 10 Datensätze für einen stratifizierten Split anlegen
    curated_data = []
    benchmark_data = []
    
    for i in range(10):
        cid = f"CALL_{i:03d}"
        curated_data.append({"call_id": cid, "text": f"Dialog Text {i}"})
        benchmark_data.append({"call_id": cid, "ground_truth": "fraud" if i % 2 == 0 else "legit"})
        
    with open(curated_input, "w", encoding="utf-8") as f:
        for d in curated_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    with open(benchmark_input, "w", encoding="utf-8") as f:
        for d in benchmark_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    with patch("src.customizer.prepare_sft.CURATED_INPUT", str(curated_input)), \
         patch("src.customizer.prepare_sft.BENCHMARK_INPUT", str(benchmark_input)), \
         patch("src.customizer.prepare_sft.SFT_DIR", str(sft_dir)):
        
        prepare_sft_splits()
        
        train_file = sft_dir / "train.jsonl"
        val_file = sft_dir / "validation.jsonl"
        test_file = sft_dir / "test.jsonl"
        
        assert train_file.exists()
        assert val_file.exists()
        assert test_file.exists()
        
        # Prüfen, ob NeMo Customizer Format ("input" / "output") eingehalten wird
        with open(train_file, "r", encoding="utf-8") as f:
            sample = json.loads(f.readline())
            assert "input" in sample
            assert "output" in sample
            assert sample["output"] in ["fraud", "legit"]