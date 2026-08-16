import json
import re
import requests
from sklearn.metrics import classification_report, accuracy_score

VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation_new.jsonl"
NIM_URL = "http://localhost:8000/v1/chat/completions"

def evaluate_model(mode="zero-shot"):
    print(f"\n🚀 Starte Evaluierung im Modus: {mode.upper()} ...\n")
    
    y_true = []
    y_pred = []
    total = 0
    
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            transcript_text = data.get("input", "")
            true_label = data.get("output", "").strip().lower()
            
            if not transcript_text:
                continue
            
            # Nachrichten-Struktur je nach Modus anpassen (Zero-Shot vs Few-Shot ICL)
            messages = [
                {
                    "role": "system",
                    "content": "Du bist ein präziser Betrugserkennungs-Assistent. Analysiere das Transkript und antworte am Ende mit exakt 'fraud' oder 'legitimate'."
                }
            ]
            
            if mode == "few-shot":
                # In-Context Learning (ICL) Beispiele hinzufügen
                messages.extend([
                    {
                        "role": "user",
                        "content": "Kunde: Hallo, ich wollte nur fragen, wann meine nächste Rechnung abgebucht wird. Agent: Guten Tag, die Abbuchung erfolgt am Monatsletzten."
                    },
                    {
                        "role": "assistant",
                        "content": "legitimate"
                    },
                    {
                        "role": "user",
                        "content": "Kunde: Mein Konto ist komplett leergeräumt, geben Sie sofort die TAN frei!"
                    },
                    {
                        "role": "assistant",
                        "content": "fraud"
                    }
                ])
            
            # Aktuelles Transkript anhängen
            messages.append({
                "role": "user",
                "content": transcript_text
            })
            
            payload = {
                "model": "meta/llama-3.1-8b-instruct",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 50
            }
            
            try:
                response = requests.post(NIM_URL, json=payload)
                result = response.json()
                model_raw = result["choices"][0]["message"]["content"].strip().lower()
                
                # Robustes Parsing
                if "legitimate" in model_raw and "fraud" not in model_raw:
                    model_answer = "legitimate"
                elif "fraud" in model_raw and "legitimate" not in model_raw:
                    model_answer = "fraud"
                else:
                    match = re.search(r'\b(fraud|legitimate)\b', model_raw)
                    model_answer = match.group(1) if match else "unknown"
                
                y_true.append(true_label)
                y_pred.append(model_answer)
                total += 1
                
            except Exception as e:
                print(f"Fehler bei Eintrag {total}: {e}")

    # Ergebnisse ausgeben
    acc = accuracy_score(y_true, y_pred)
    print(f"📊 ERGEBNISSE FÜR {mode.upper()}:")
    print(f"Gesamtgenauigkeit (Accuracy): {acc * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_true, y_pred, labels=["fraud", "legitimate"], target_names=["fraud", "legitimate"], zero_division=0))

if __name__ == "__main__":
    # Beide Modi nacheinander ausführen zum direkten Vergleich (wie im Notebook)
    evaluate_model(mode="zero-shot")
    evaluate_model(mode="few-shot")