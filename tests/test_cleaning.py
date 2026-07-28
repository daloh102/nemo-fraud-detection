import os
import json
import pytest
from unittest.mock import patch
from src.curator.cleaning import mask_pii_text, run_cleaning_pipeline

def test_mask_pii_text():
    """Testet direkte Regex-Maskierungen und UTF-8 Repair."""
    # 1. IBAN
    raw_iban = "Meine IBAN ist DE89 3704 0044 0532 0130 00."
    assert "[IBAN_MASKIERT]" in mask_pii_text(raw_iban)
    
    # 2. Kreditkarte
    raw_cc = "Karte: 4532-1100-8821-9943"
    assert "[KREDITKARTE_MASKIERT]" in mask_pii_text(raw_cc)
    
    # 3. Umlaute
    broken_encoding = "SchÃ¶ner Tag in MÃ¼nchen"
    assert mask_pii_text(broken_encoding) == "Schöner Tag in München"

def test_run_cleaning_pipeline(tmp_path):
    # Test-Eingabedaten erzeugen (Inklusive Kurztext/Müll)
    input_file = tmp_path / "raw_test.jsonl"
    interim_cleaned = tmp_path / "cleaned_calls.jsonl"
    interim_pii = tmp_path / "pii_removed_calls.jsonl"
    
    test_data = [
        {"call_id": "CALL_001", "text": "Guten Tag, ich möchte meine IBAN DE89 3704 0044 0532 0130 00 anpassen lassen."}, # PII
        {"call_id": "CALL_002", "text": "OK"}, # Zu kurz (< 50 Zeichen) -> muss gefiltert werden
        {"call_id": "CALL_003", "text": "Hallo Kundenservice, hier ist ein ausreichend langer Testdialog für die Pipeline."}
    ]
    
    with open(input_file, "w", encoding="utf-8") as f:
        for d in test_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    with patch("src.curator.cleaning.INPUT_RAW", str(input_file)), \
         patch("src.curator.cleaning.INTERIM_CLEANED", str(interim_cleaned)), \
         patch("src.curator.cleaning.INTERIM_PII_REMOVED", str(interim_pii)):
        
        run_cleaning_pipeline()
        
        assert interim_cleaned.exists()
        
        with open(interim_cleaned, "r", encoding="utf-8") as f:
            cleaned_docs = [json.loads(l) for l in f]
            
        # "OK" (CALL_002) muss gefiltered worden sein -> Es bleiben 2 Dokumente
        assert len(cleaned_docs) == 2
        assert "[IBAN_MASKIERT]" in cleaned_docs[0]["text"]