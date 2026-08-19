"""
================================================================================
Projekt:        NeMo Fraud Detection
Skript-Name:    run_llm_judge.py
Beschreibung:   Erweiterte LLM-gestützte Evaluierung (LLM-as-a-Judge) mit strukturierter 
                Chain-of-Thought-Analyse für Kundengespräche und Betrugserkennung.

Funktionsumfang:
    1. Strukturierte Analyse (Worker-Modell): Das Modell analysiert Transkripte 
       strikt nach Emotionen/Druck, Sicherheitsrisiko und finalem Resultat.
    2. Automatisierter Qualitätsprüfer (Judge-Modell): Ein zweiter Prompt prüft 
       unabhängig, ob die Analyse inhaltlich zum erwarteten Label (Ground Truth) passt.
    3. Tabellarische Aufbereitung: Sammelt alle Ergebnisse in einem Pandas-DataFrame 
       und gibt eine formatierte Übersicht im Terminal aus.
    4. Reporting: Exportiert den detaillierten Audit-Report inklusive Begründungen 
       als CSV-Datei zur weiteren Verwendung.

Eingabedateien:
    - Validierungsdaten: /data/nemo-fraud-detection/data/sft/validation.jsonl

Ausgabedateien:
    - CSV-Report: /data/nemo-fraud-detection/data/evaluation/evaluation_report.csv

Autor:          Daniel Lohmann
Jahr:           2026
Erfolgreich getestet am: 19.08.2026
================================================================================
"""
import json
import re
import requests
import pandas as pd

VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation.jsonl"
NIM_URL = "http://localhost:8800/v1/chat/completions"

def evaluate_with_llm_judge():
    print(f"🚀 Starte Evaluierung mit strukturierter Pandas-Auswertung...\n")
    
    results_list = []
    total = 0
    passed = 0
    
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            transcript_text = data.get("input", "")
            true_label = data.get("output", "").strip().lower()
            
            if not transcript_text:
                continue
            
            total += 1
            
            # 1. Schritt: Das Arbeitsmodell
            worker_payload = {
                "model": "meta/llama-3.1-8b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Du bist ein Experte für Betrugserkennung in Kundengesprächen. "
                            "Analysiere das Transkript strikt nach folgendem Format:\n"
                            "1. Emotionen/Druck: [Beschreibung]\n"
                            "2. Sicherheitsrisiko: [Beschreibung]\n"
                            "Result: [fraud oder legitimate]"
                        )
                    },
                    {
                        "role": "user",
                        "content": "Kunde: Hallo, ich wollte nur fragen, wann meine nächste Rechnung abgebucht wird. Agent: Guten Tag, die Abbuchung erfolgt immer am Monatsletzten."
                    },
                    {
                        "role": "assistant",
                        "content": "1. Emotionen/Druck: Ruhig, normal, kein Druck.\n2. Sicherheitsrisiko: Keins, einfache Rechnungsanfrage.\nResult: legitimate"
                    },
                    {
                        "role": "user",
                        "content": "Kunde: Mein Konto ist komplett leergeräumt, geben Sie sofort die TAN frei!"
                    },
                    {
                        "role": "assistant",
                        "content": "1. Emotionen/Druck: Extrem panisch, starker Druck auf den Agenten zur sofortigen Freigabe.\n2. Sicherheitsrisiko: Umgehung von Sicherheitsstandards, Betrugsverdacht.\nResult: fraud"
                    },
                    {
                        "role": "user",
                        "content": transcript_text
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 150
            }
            
            try:
                worker_response = requests.post(NIM_URL, json=worker_payload)
                worker_result = worker_response.json()
                model_output = worker_result["choices"][0]["message"]["content"].strip()
                
                # Extrahieren des Results aus dem Modell-Output
                if "result: legitimate" in model_output.lower() and "fraud" not in model_output.lower():
                    model_answer = "legitimate"
                elif "result: fraud" in model_output.lower() and "legitimate" not in model_output.lower():
                    model_answer = "fraud"
                else:
                    match = re.search(r'\b(fraud|legitimate)\b', model_output.lower())
                    model_answer = match.group(1) if match else "unknown"
                
                # 2. Schritt: Der LLM-Judge bewertet die Antwort
                judge_payload = {
                    "model": "meta/llama-3.1-8b-instruct",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Du bist ein strenger Qualitätsprüfer für KI-gestützte Betrugserkennung. "
                                "Prüfe, ob das KI-Modell in seiner Analyse zum selben 'Result' wie das erwartete Label (Ground Truth) kommt.\n"
                                "Antworte ausschliesslich in diesem Format:\n"
                                "Score: PASS (falls korrekt) oder Score: FAIL (falls falsch)\n"
                                "Begründung: [Kurzer Text]"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Transkript: {transcript_text}\n"
                                f"Erwartetes Label (Ground Truth): {true_label}\n"
                                f"KI-Modell Antwort:\n{model_output}"
                            )
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 100
                }
                
                judge_response = requests.post(NIM_URL, json=judge_payload)
                judge_result = judge_response.json()
                judge_output = judge_result["choices"][0]["message"]["content"].strip()
                
                is_pass = "score: pass" in judge_output.lower()
                if is_pass:
                    passed += 1
                
                # Zeile für den DataFrame speichern
                results_list.append({
                    "id": total,
                    "true_label": true_label,
                    "model_prediction": model_answer,
                    "judge_score": "PASS" if is_pass else "FAIL",
                    "model_output": model_output.replace("\n", " ")
                })
                
                print(f"Eintrag {total} | Judge: {'✅ PASS' if is_pass else '❌ FAIL'}")
                
            except Exception as e:
                print(f"Fehler bei Eintrag {total}: {e}")

    # DataFrame aus den Ergebnissen erzeugen (angelehnt an dein Pandas-Snippet)
    df = pd.DataFrame(results_list)
    
    print("\n" + "="*80)
    print("📊 PANDAS EVALUIERUNGS-TABELLE (AUSZUG)")
    print("="*80)
    # Zeige die ersten Einträge als strukturierte Tabelle
    print(df[["id", "true_label", "model_prediction", "judge_score"]].to_string(index=False))
    
    success_rate = (passed / total) * 100 if total > 0 else 0
    print("\n" + "="*50)
    print(f"✅ Erfolgreich bewertet (PASS): {passed} / {total}")
    print(f"🎯 Finale Judge-Zufriedenheitsrate: {success_rate:.2f}%")
    print("="*50)

    # Optional: Als CSV abspeichern für spätere Berichte
    df.to_csv("/data/nemo-fraud-detection/data/evaluation/evaluation_report.csv", index=False)
    print("📁 Ausführlicher Report gespeichert unter: /data/nemo-fraud-detection/data/evaluation/evaluation_report.csv")

if __name__ == "__main__":
    evaluate_with_llm_judge()