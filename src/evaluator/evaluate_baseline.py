import json
import re
import requests
import wandb

VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation_new.jsonl"
NIM_URL = "http://localhost:8000/v1/chat/completions"  # Nutzt den Chat-Endpunkt

def run_chat_baseline():
    wandb.init(project="fraud-detection", name="baseline-chat-evaluation")
    print(f"🚀 Starte faire Chat-Baseline-Evaluierung mit dem Basismodell...\n")
    
    correct = 0
    total = 0
    
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            input_content = data.get("input", "")
            true_label = data.get("output", "").strip().lower()
            
            if not input_content:
                continue
                
            payload = {
                "model": "meta-llama/Llama-3.2-3B-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "Du bist ein Betrugserkennungs-Assistent für Kundengespräche. Analysiere das Transkript und antworte AUSSCHLIESSLICH mit exakt einem Wort: 'fraud' oder 'legitimate'."
                    },
                    {
                        "role": "user",
                        "content": input_content
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 10
            }
            
            try:
                response = requests.post(NIM_URL, json=payload)
                result = response.json()
                
                if "choices" not in result:
                    print(f"Fehler von API bei Eintrag {total + 1}: {result}")
                    continue

                # Antwort aus der Chat-Struktur auslesen
                model_raw = result["choices"][0]["message"]["content"].strip().lower()
                
                # Robustes Parsing via Regex
                if "legitimate" in model_raw and "fraud" not in model_raw:
                    model_answer = "legitimate"
                elif "fraud" in model_raw and "legitimate" not in model_raw:
                    model_answer = "fraud"
                else:
                    match = re.search(r'\b(fraud|legitimate)\b', model_raw)
                    model_answer = match.group(1) if match else model_raw
                
                is_correct = (true_label == model_answer)
                if is_correct:
                    correct += 1
                total += 1
                
                print(f"Eintrag {total} | Erwartet: {true_label} | Erkannt: {model_answer} | Korrekt: {is_correct}")
                
            except Exception as e:
                print(f"Fehler bei Eintrag {total + 1}: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    
    wandb.log({
        "eval/accuracy": accuracy,
        "eval/correct": correct,
        "eval/total": total
    })
    wandb.finish()

    print("\n" + "="*40)
    print("📊 CHAT-BASELINE EVALUIERUNGSERGEBNISSE")
    print("="*40)
    print(f"✅ Korrekt erkannt: {correct} / {total}")
    print(f"🎯 Genauigkeit:     {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_chat_baseline()