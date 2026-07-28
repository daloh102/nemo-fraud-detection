import os
import json
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CURATED_INPUT = os.path.join(BASE_DIR, "data", "curated", "deduplicated_calls.jsonl")
BENCHMARK_INPUT = os.path.join(BASE_DIR, "data", "evaluation", "benchmark_dataset.jsonl")

SFT_DIR = os.path.join(BASE_DIR, "data", "sft")

def prepare_sft_splits():
    print("📊 Bereite SFT Datensätze vor...")
    os.makedirs(SFT_DIR, exist_ok=True)
    
    # 1. Benchmarks / Ground Truth Mapping laden
    ground_truth_map = {}
    with open(BENCHMARK_INPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ground_truth_map[item["call_id"]] = item["ground_truth"]
                
    # 2. Kuratierte Transkripte laden & matchen
    sft_records = []
    with open(CURATED_INPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                cid = doc.get("call_id")
                # Nur Datensätze verwenden, die in der Benchmark-Truth existieren
                if cid in ground_truth_map:
                    label = ground_truth_map[cid]
                    # Formatierung für NeMo Customizer SFT
                    sft_payload = {
                        "input": f"Klassifiziere folgendes Telefongespräch als 'fraud' oder 'legit':\n{doc['text']}\nKlassifikation:",
                        "output": label
                    }
                    sft_records.append(sft_payload)

    # 3. Stratifizierter Split (70% Train / 15% Val / 15% Test)
    labels = [r["output"] for r in sft_records]
    train_data, rest_data, _, rest_labels = train_test_split(
        sft_records, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_data, test_data = train_test_split(
        rest_data, test_size=0.50, stratify=rest_labels, random_state=42
    )
    
    def save_file(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"   -> Exportiert: {path} ({len(data)} Einträge)")

    save_file(os.path.join(SFT_DIR, "train.jsonl"), train_data)
    save_file(os.path.join(SFT_DIR, "validation.jsonl"), val_data)
    save_file(os.path.join(SFT_DIR, "test.jsonl"), test_data)

    print("✅ SFT Datensätze erfolgreich im Ordner 'data/sft/' gespeichert!")

if __name__ == "__main__":
    prepare_sft_splits()