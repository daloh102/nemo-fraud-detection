import os
import json
import pytest
from unittest.mock import patch
from src.curator.deduplication import run_deduplication

def test_run_deduplication(tmp_path):
    input_interim = tmp_path / "cleaned_calls.jsonl"
    output_curated = tmp_path / "deduplicated_calls.jsonl"
    
    # 3 Dokumente, wovon 2 identischen Text enthalten
    test_docs = [
        {"call_id": "CALL_001", "text": "Dies ist ein eindeutiger Testtext für die Deduplizierung."},
        {"call_id": "CALL_001_DUP", "text": "Dies ist ein eindeutiger Testtext für die Deduplizierung."}, # Duplikat!
        {"call_id": "CALL_002", "text": "Dies ist ein völlig anderer Text für die Deduplizierung."}
    ]
    
    with open(input_interim, "w", encoding="utf-8") as f:
        for d in test_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    with patch("src.curator.deduplication.INPUT_INTERIM", str(input_interim)), \
         patch("src.curator.deduplication.CURATED_DEDUP", str(output_curated)):
        
        run_deduplication()
        
        assert output_curated.exists()
        
        with open(output_curated, "r", encoding="utf-8") as f:
            result_docs = [json.loads(l) for l in f]
            
        # Duplikat muss entfernt worden sein -> Nur 2 eindeutige Dokumente
        assert len(result_docs) == 2