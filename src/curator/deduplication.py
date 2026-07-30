import os
import json
from nemo_curator.datasets import DocumentDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_INTERIM = os.path.join(BASE_DIR, "data", "interim", "cleaned_calls.jsonl")
CURATED_DEDUP = os.path.join(BASE_DIR, "data", "curated", "deduplicated_calls.jsonl")

def run_deduplication():
    print(f"🔍 Starte Deduplizierung aus: {INPUT_INTERIM}")
    os.makedirs(os.path.dirname(CURATED_DEDUP), exist_ok=True)
    
    dataset = DocumentDataset.read_json(INPUT_INTERIM)
    docs = dataset.to_dict()
    
    seen_hashes = set()
    unique_docs = []
    
    for doc in docs:
        text_content = doc.get("text", "").strip()
        text_hash = hash(text_content)
        
        if text_hash not in seen_hashes and len(text_content) > 0:
            seen_hashes.add(text_hash)
            unique_docs.append(doc)
            
    print(f"💾 Schreibe deduplizierte Daten nach: {CURATED_DEDUP}")
    with open(CURATED_DEDUP, "w", encoding="utf-8") as f:
        for doc in unique_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    print(f"✅ Deduplizierung abgeschlossen! ({len(unique_docs)} einzigartige Dokumente)")

if __name__ == "__main__":
    run_deduplication()