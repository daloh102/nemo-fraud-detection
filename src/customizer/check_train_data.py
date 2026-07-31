import json

TRAIN_FILE = "/data/nemo-fraud-detection/data/sft/train.jsonl"

def analyze_train_data():
    print(f"📂 Analysiere Trainingsdatei: {TRAIN_FILE}\n")
    
    fraud_count = 0
    legit_count = 0
    other_count = 0

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            example = json.loads(line)
            
            # Überprüfe den Output oder andere relevante Felder
            output_text = str(example.get("output", "")).lower()
            input_text = str(example.get("input", "")).lower()
            
            # Suche nach Markierungen für Fraud oder Legitimate
            content = output_text + " " + input_text
            
            if "fraud" in content and "legitimate" not in content:
                fraud_count += 1
            elif "legitimate" in content or "legitim" in content:
                legit_count += 1
            else:
                other_count += 1

    total = fraud_count + legit_count + other_count

    print("="*40)
    print("📊 KLASSEN-VERTEILUNG IM TRAININGSDATENSATZ")
    print("="*40)
    print(f"🚨 Fraud-Fälle:     {fraud_count} ({ (fraud_count/total)*100:.1f}% if total > 0 else 0) %")
    print(f"✅ Legitimate-Fälle: {legit_count} ({ (legit_count/total)*100:.1f}% if total > 0 else 0) %")
    if other_count > 0:
        print(f"❓ Unzuzuordnen:   {other_count}")
    print(f"📈 Gesamtanzahl:     {total}")
    print("="*40)

if __name__ == "__main__":
    analyze_train_data()