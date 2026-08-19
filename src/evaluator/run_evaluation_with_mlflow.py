"""
================================================================================
Projekt:        NeMo Fraud Detection
Skript-Name:    run_evaluation_with_mlflow.py
Beschreibung:   Evaluierung des Basismodells (Llama-3.1-8B-Instruct via NIM)
                in den Modi 'Zero-Shot' und 'Few-Shot' zur Betrugserkennung.
                Berechnet Metriken wie Accuracy, Precision, Recall und F1-Score
                und protokolliert die Ergebnisse automatisch in MLflow.

Funktionsumfang:
    1. Modus-Steuerung: Testet das Modell wahlweise mit reiner Aufgabenstellung 
       (Zero-Shot) oder inkl. Positiv-/Negativ-Beispielen (Few-Shot).
    2. API-Kommunikation: Fragt den lokalen NIM-Endpunkt per Chat-Completion ab.
    3. Robustes Parsing: Extrahiert das finale Ergebnis ('fraud' / 'legitimate') 
       aus der generierten Textantwort.
    4. Metrik-Berechnung & Tracking: Nutzt scikit-learn für detaillierte Metriken 
       und speichert Runs direkt in MLflow.

Eingabedateien:
    - Validierungsdaten: /data/nemo-fraud-detection/data/sft/validation.jsonl

Autor:          Daniel Lohmann
Jahr:           2026
Erfolgreich getstet am: 19.08.2026
================================================================================
"""
import json
import re
import requests
import mlflow
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation.jsonl"
NIM_URL = "http://localhost:8800/v1/chat/completions"

# MLflow Experiment konfigurieren
mlflow.set_experiment("NeMo-Fraud-Detection-Evaluation")

def evaluate_and_log(mode="few-shot"):
    print(f"\n🚀 Starte Evaluierung im Modus: {mode.upper()} mit MLflow Tracking...\n")
    
    y_true = []
    y_pred = []
    total = 0
    passed_judge = 0
    
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            transcript_text = data.get("input", "")
            true_label = data.get("output", "").strip().lower()
            
            if not transcript_text:
                continue
            
            total += 1
            
            # Messages je nach Modus aufbauen
            messages = [
                {
                    "role": "system",
                    "content": "Du bist ein Experte für Betrugserkennung. Analysiere das Transkript kurz, begründe deine Entscheidung und beende deine Antwort mit exakt 'Result: fraud' oder 'Result: legitimate'."
                }
            ]
            
            if mode == "few-shot":
                messages.extend([
                    {
                        "role": "user",
                        "content": "Kunde: Hallo, ich wollte nur fragen, wann meine nächste Rechnung abgebucht wird. Agent: Guten Tag, die Abbuchung erfolgt immer am Monatsletzten."
                    },
                    {
                        "role": "assistant",
                        "content": "Analyse: Normale Anfrage zur Rechnung ohne Betrugsindizien. Result: legitimate"
                    },
                    {
                        "role": "user",
                        "content": "Kunde: Mein Konto ist komplett leergeräumt, geben Sie sofort die TAN frei!"
                    },
                    {
                        "role": "assistant",
                        "content": "Analyse: Panikmache und Druck auf den Agenten zur TAN-Freigabe. Klassischer Betrug. Result: fraud"
                    }
                ])
            
            messages.append({"role": "user", "content": transcript_text})
            
            payload = {
                "model": "meta/llama-3.1-8b-instruct",
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 150
            }
            
            try:
                response = requests.post(NIM_URL, json=payload)
                result = response.json()
                model_output = result["choices"][0]["message"]["content"].strip().lower()
                
                # Parsing des Ergebnisses
                if "result: legitimate" in model_output and "fraud" not in model_output:
                    model_answer = "legitimate"
                elif "result: fraud" in model_output and "legitimate" not in model_output:
                    model_answer = "fraud"
                else:
                    match = re.search(r'\b(fraud|legitimate)\b', model_output)
                    model_answer = match.group(1) if match else "unknown"
                
                y_true.append(true_label)
                y_pred.append(model_answer)
                
                # Einfacher Check ob Label übereinstimmt für provisorischen Judge
                if model_answer == true_label:
                    passed_judge += 1
                    
            except Exception as e:
                print(f"Fehler bei Eintrag {total}: {e}")

    # Metriken berechnen
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["fraud", "legitimate"], average="weighted", zero_division=0
    )
    judge_rate = (passed_judge / total) * 100 if total > 0 else 0

    print(f"📊 ERGEBNISSE FÜR {mode.upper()}:")
    print(f"Accuracy: {acc * 100:.2f}% | F1-Score: {f1:.4f} | Match-Rate: {judge_rate:.2f}%")
    print("-" * 40)
    print(classification_report(y_true, y_pred, labels=["fraud", "legitimate"], target_names=["fraud", "legitimate"], zero_division=0))

    # In MLflow einloggen
    with mlflow.start_run(run_name=f"evaluation-{mode}"):
        mlflow.log_param("evaluation_mode", mode)
        mlflow.log_param("model_name", "meta/llama-3.1-8b-instruct")
        mlflow.log_param("dataset_file", VAL_FILE)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("weighted_f1", f1)
        mlflow.log_metric("weighted_precision", precision)
        mlflow.log_metric("weighted_recall", recall)
        mlflow.log_metric("match_rate_percentage", judge_rate)
        
        print(f"✅ Ergebnisse erfolgreich an MLflow übergeben für Run-Modus: {mode}\n")

if __name__ == "__main__":
    # Beide Modi nacheinander tracken, um sie in MLflow direkt zu vergleichen
    evaluate_and_log(mode="zero-shot")
    evaluate_and_log(mode="few-shot")