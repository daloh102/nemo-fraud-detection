import os
import sys
import json
import random
from pathlib import Path

# ==============================================================================
# 1. PFAD-KONFIGURATION
# ==============================================================================
BASE_DIR = Path("/data/nemo-fraud-detection")
DATA_DIR = BASE_DIR / "data"

CURATED_PATH = DATA_DIR / "curated" / "dialogues_transcripts_curator.jsonl"
BENCHMARK_PATH = DATA_DIR / "evaluation" / "dialogues_benchmark.jsonl"
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

    # 1. Lookup-Map aus Benchmark aufbauen (Schlüssel: "id", z.B. "doc-00001")
    gt_map = {}
    for item in benchmark_data:
        cid = item.get("id")
        label = item.get("label") or item.get("fraud_type")
        if cid and label:
            gt_map[str(cid)] = label

    # 2. Matching über 'call_id' ausführen
    sft_samples = []
    matched_count = 0

    for doc in curated_data:
        text = doc.get("text", "")
        # Priorisiere 'call_id' (z. B. "doc-00001"), Fallback auf 'id'
        cid = str(doc.get("call_id") or doc.get("id", ""))
        
        gt_label = gt_map.get(cid)
        
        if gt_label:
            matched_count += 1
            sft_samples.append({
                "input": text,
                "output": gt_label
            })

    total_docs = len(curated_data)
    print(f"ℹ️ Total geladene Dokumente: {total_docs}")
    print(f"   Davon erfolgreich mit Ground-Truth gematcht: {matched_count}")

    # STRIKTER ABBRUCH: Bricht ab, falls auch nur 1 Dokument nicht gematcht werden kann
    if matched_count < total_docs:
        print(f"\n❌ KRITISCHER MATCHING-FEHLER: Es konnten nur {matched_count} von {total_docs} Dokumenten gematcht werden!")
        print("   -> Der Vorgang wird abgebrochen, um unvollständige SFT-Daten zu verhindern.")
        sys.exit(1)

    # 3. Train / Val / Test Split (70% / 15% / 15%)
    random.seed(42)
    random.shuffle(sft_samples)

    train_end = int(total_docs * 0.70)
    val_end = train_end + int(total_docs * 0.15)

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

    print(f"\n   -> Exportiert: train.jsonl ({len(train_data)} Einträge)")
    print(f"   -> Exportiert: validation.jsonl ({len(val_data)} Einträge)")
    print(f"   -> Exportiert: test.jsonl ({len(test_data)} Einträge)")
    print(f"\n✅ SFT-Datensätze mit 100% Matching-Quote in '{SFT_DIR}' gespeichert!")

if __name__ == "__main__":
    main()