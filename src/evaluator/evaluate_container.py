import json
import re
import requests

VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation_new.jsonl"
NIM_URL = "http://localhost:8000/v1/completions"  # Korrekter Completions-Endpunkt

def run_evaluation():
    print(f"🚀 Starte Evaluierung über den vLLM Completions-Endpunkt...\n")
    
    correct = 0
    total = 0
    
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            input_content = data.get("input", "")
            true_label = data.get("output", "").strip().lower()
            
            if not input_content:
                continue
                
            # Exaktes Trainings-Template als einzelner String
            prompt_text = f"### Instruction:\n{input_content}\n\n### Response:\n"
            
            payload = {
                "model": "fraud-detection",
                "prompt": prompt_text,
                "temperature": 0.1,
                "max_tokens": 10
            }
            
            try:
                response = requests.post(NIM_URL, json=payload)
                result = response.json()
                
                if "choices" not in result:
                    print(f"Fehler von API bei Eintrag {total + 1}: {result}")
                    continue

                # Bei /v1/completions liegt der Text direkt unter "text"
                model_raw = result["choices"][0]["text"].strip().lower()
                
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
    print("\n" + "="*40)
    print("📊 FINALE EVALUIERUNGSERGEBNISSE")
    print("="*40)
    print(f"✅ Korrekt erkannt: {correct} / {total}")
    print(f"🎯 Genauigkeit:     {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()