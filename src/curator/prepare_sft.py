"""
Projekt:        NeMo Fraud Detection
Skript-Name:    sft_data_preparation.py
Beschreibung:   Vorbereitung und Aufteilung von Supervised Fine-Tuning (SFT) 
                Datensätzen für das Betrugserkennungs-Modell (Fraud Detection).

Funktionsumfang:
    1. Validierung: Prüft das Vorhandensein aller benötigten Eingabedateien 
       (Fail-Fast-Prinzip).
    2. Matching: Verknüpft kuratierte Dialog-Transkripte über eine eindeutige 
       ID ('call_id' / 'id') mit den entsprechenden Ground-Truth-Labels aus dem Benchmark.
    3. Data Splitting: Führt einen reproduzierbaren Train / Validation / Test 
       Split im Verhältnis 70% / 15% / 15% durch (Seed: 42).
    4. Export: Speichert die aufbereiteten Datensätze als JSONL-Dateien 
       im SFT-Zielverzeichnis.

Eingabedateien:
    - Kuratierte Transkripte: /data/nemo-fraud-detection/data/curated/transcripts_curated.jsonl
    - Benchmark-Labels:      /data/nemo-fraud-detection/data/evaluation/transcripts_benchmark.jsonl

Ausgabedateien (im Verzeichnis /data/nemo-fraud-detection/data/sft/):
    - train.jsonl       (Training Data - 70%)
    - validation.jsonl  (Validation Data - 15%)
    - test.jsonl        (Test Data - 15%)

Autor:         Daniel Lohmann
Datum:         2026
Erfolgreich getestet am: 19.08.2026
"""

import sys
import json
import random
from pathlib import Path

# ==============================================================================
# 1. PFAD-KONFIGURATION
# ==============================================================================
BASE_DIR = Path("/data")
DATA_DIR = BASE_DIR / "nemo-fraud-detection" / "data"

CURATED_PATH = DATA_DIR / "curated" / "transcripts_curated.jsonl"
# Korrigierter Dateiname hier:
BENCHMARK_PATH = DATA_DIR / "evaluation" / "transcripts_benchmark.jsonl"
SFT_DIR = DATA_DIR / "sft"

# ==============================================================================
# 2. STRIKTES ERROR HANDLING (FAIL-FAST)
# ==============================================================================
def validate_required_files():
    print("🔍 Prüfe Eingabedateien...")
    
    if not CURATED_PATH.exists():
        print(f"❌ KRITISCHER FEHLER: Kuratierte Datei nicht gefunden: '{CURATED_PATH}'")
        sys.exit(1)
        
    if not BENCHMARK_PATH.exists():
        print(f"❌ KRITISCHER FEHLER: Benchmark-Datei nicht gefunden: '{BENCHMARK_PATH}'")
        sys.exit(1)

    print("✅ Alle benötigten Dateien wurden gefunden.\n")

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
def main():
    validate_required_files()

    print("📊 Bereite SFT-Datensätze vor...")
    
    curated_data = load_jsonl(CURATED_PATH)
    benchmark_data = load_jsonl(BENCHMARK_PATH)

    # 1. Lookup-Map aus Benchmark aufbauen (Schlüssel: "id")
    gt_map = {}
    for item in benchmark_data:
        cid = item.get("id")
        label = item.get("label") or item.get("fraud_type")
        if cid and label:
            gt_map[str(cid)] = label

    # 2. Matching über 'call_id' oder 'id' ausführen
    sft_samples = []
    matched_count = 0
    unmatched_count = 0

    for doc in curated_data:
        text = doc.get("text", "")
        cid = str(doc.get("call_id") or doc.get("id", ""))
        
        gt_label = gt_map.get(cid)
        
        if gt_label:
            matched_count += 1
            sft_samples.append({
                "input": text,
                "output": gt_label
            })
        else:
            unmatched_count += 1

    total_docs = len(curated_data)
    print(f"ℹ️ Total geladene kuratierte Dokumente: {total_docs}")
    print(f"    🔗 Erfolgreich mit Ground-Truth gematcht: {matched_count}")
    
    if unmatched_count > 0:
        print(f"    ⚠️ Ohne passendes Label übersprungen: {unmatched_count} Dokument(e)")

    if len(sft_samples) == 0:
        print(f"\n❌ KRITISCHER FEHLER: Es konnten keine einzigen SFT-Daten gematcht werden!")
        sys.exit(1)

    # 3. Train / Val / Test Split (70% / 15% / 15%)
    random.seed(42)
    random.shuffle(sft_samples)

    total_samples = len(sft_samples)
    train_end = int(total_samples * 0.70)
    val_end = train_end + int(total_samples * 0.15)

    train_data = sft_samples[:train_end]
    val_data = sft_samples[train_end:val_end]
    test_data = sft_samples[val_end:]

    # 4. Speichern
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_path = SFT_DIR / "train.jsonl"
    val_path = SFT_DIR / "validation.jsonl"
    test_path = SFT_DIR / "test.jsonl"

    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    save_jsonl(test_data, test_path)

    print(f"\n    -> Exportiert: train.jsonl ({len(train_data)} Einträge)")
    print(f"    -> Exportiert: validation.jsonl ({len(val_data)} Einträge)")
    print(f"    -> Exportiert: test.jsonl ({len(test_data)} Einträge)")
    print(f"\n✅ SFT-Datensätze erfolgreich in '{SFT_DIR}' gespeichert!")

if __name__ == "__main__":
    main()