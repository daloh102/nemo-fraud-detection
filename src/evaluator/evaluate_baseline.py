"""
================================================================================
Projekt:        NeMo Fraud Detection
Skript-Name:    evaluate_baseline.py
Beschreibung:   Faire Chat-Baseline-Evaluierung des unmodifizierten Basismodells 
                (Llama-3.1-8B-Instruct via NIM) vor dem Supervised Fine-Tuning (SFT).

Funktionsumfang:
    1. Pre-Flight-Check: Prüft vorab, ob der NIM-Service erreichbar ist (Fail-Fast).
    2. API-Abfrage: Sendet validierte Transkripte per Chat-Completion-Endpunkt an das Modell.
    3. Label-Mapping & Parsing: Mappt Ground-Truth-Labels und extrahiert die 
       Modellantwort robust via Regex.
    4. Metriken & Tracking: Berechnet die Genauigkeit (Accuracy) und loggt 
       die Ergebnisse automatisch an Weights & Biases (wandb).

Eingabedateien:
    - Validierungsdaten: /data/nemo-fraud-detection/notebooks/02_Data_Curation/data/sft/validation.jsonl

Autor:         Daniel Lohmann
Datum:         2026
Erfolgreich getestet am: 19.08.2026
================================================================================
"""
import json
import re
import requests
import wandb
import sys

VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation.jsonl"
# WICHTIG: Falls der NIM auf dem Host läuft und das Skript im Container,
# nutze ggf. die IP des Host-Netzwerks statt 'localhost'
NIM_URL = "http://localhost:8800/v1/chat/completions"

def check_llm_connection():
    """Prüft vor dem Start, ob der NIM-Service erreichbar ist."""
    print("🔍 Prüfe Verbindung zum LLM (NIM-Service)...")
    try:
        # Ein einfacher Test-Call an den NIM
        response = requests.get("http://localhost:8800/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ Verbindung zum LLM steht!")
            return True
        else:
            print(f"❌ Verbindung fehlgeschlagen. Status-Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Verbindung zum NIM-Service unter '{NIM_URL}' nicht möglich: {e}")
        print("💡 Tipp: Wenn das Skript im Docker läuft, stelle sicher, dass der NIM erreichbar ist (ggf. --network host).")
        return False

def run_chat_baseline():
    if not check_llm_connection():
        sys.exit(1)

    wandb.init(project="fraud-detection", name="baseline-chat-evaluation")
    print(f"🚀 Starte faire Chat-Baseline-Evaluierung...\n")
    
    correct = 0
    total = 0
    
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            input_content = data.get("input", "")
            raw_true = data.get("output", "").strip().lower()
            
            # Mappe deutsche Labels auf das englische Schema ('fraud' / 'legitimate')
            if "betrug" in raw_true and "kein" not in raw_true:
                true_label = "fraud"
            else:
                true_label = "legitimate"
            
            if not input_content:
                continue
                
            payload = {
                "model": "meta/llama-3.1-8b-instruct",
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