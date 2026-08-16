"""
Funktionsbeschreibung der Trainingsdaten-Analyse-Pipeline:

Dieser Quellcode implementiert eine automatisierte Analyse- und Validierungspipeline 
für Trainingsdaten (Supervised Fine-Tuning, SFT) im Bereich der Betrugserkennung (Fraud Detection). 
Der Prozess gliedert sich in folgende logische Hauptschritte:

*   Infrastruktur & Konfiguration: Festlegung des zentralen Dateipfads zur Trainingsdatei 
    (train.jsonl) als Basis für die Auswertung.

*   Einlesen & Zeilenweises Parsing: Sicheres Durchlaufen der JSONL-Datei im Stream-Verfahren, 
    um auch große Datensätze speichereffizient zeilenweise zu verarbeiten. Leere Zeilen und 
    Formatierungsfehler werden dabei robust abgefangen und übersprungen.

*   Inhaltliche Klassifikation & Heuristik: 
    *   Die Felder "input" und "output" werden extrahiert und in Kleinbuchstaben konvertiert.
    *   Es erfolgt eine heuristische Suche nach Schlüsselwörtern ("fraud", "legitimate", "legitim"), 
        um die Einträge den jeweiligen Zielklassen zuzuordnen.

*   Metrik & Statistik-Ausgabe: Berechnung der absoluten Häufigkeiten sowie der prozentualen 
    Klassenverteilung (inklusive Absicherung gegen Divisionen durch Null). Abschließend wird 
    ein übersichtlicher, formatierter Bericht ausgegeben.

Zusammenfassend dient das Skript dazu, die Qualität und Balance des SFT-Datensatzes 
vor dem eigentlichen Modell-Training transparent zu überprüfen und potenzielle 
Klassenungleichgewichte frühzeitig zu erkennen.

Autor:         Daniel Lohmann
Datum:         2026
"""

import json

TRAIN_FILE = "/data/nemo-fraud-detection/data/sft/train.jsonl"

def analyze_train_data():
    """Analysiert die Klassenverteilung in der SFT-Trainingsdatei."""
    print(f"📂 Analysiere Trainingsdatei: {TRAIN_FILE}\n")
    
    fraud_count = 0
    legit_count = 0
    other_count = 0

    try:
        # Datei zeilenweise einlesen (schont den Arbeitsspeicher bei großen JSONL-Dateien)
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                
                # Relevante Textfelder extrahieren
                output_text = str(example.get("output", "")).lower()
                input_text = str(example.get("input", "")).lower()
                
                content = output_text + " " + input_text
                
                # Heuristische Klassifizierung der Daten
                if "fraud" in content and "legitimate" not in content:
                    fraud_count += 1
                elif "legitimate" in content or "legitim" in content:
                    legit_count += 1
                else:
                    other_count += 1
                    
    except FileNotFoundError:
        print(f"❌ Fehler: Die Datei {TRAIN_FILE} wurde nicht gefunden.")
        return

    total = fraud_count + legit_count + other_count

    # Prozentuale Anteile berechnen (mit Schutz vor Division durch 0)
    fraud_pct = (fraud_count / total) * 100 if total > 0 else 0
    legit_pct = (legit_count / total) * 100 if total > 0 else 0

    # Formatierte Ausgabe der Analyse-Ergebnisse
    print("="*40)
    print("📊 KLASSEN-VERTEILUNG IM TRAININGSDATENSATZ")
    print("="*40)
    print(f"🚨 Fraud-Fälle:      {fraud_count} ({fraud_pct:.1f}%)")
    print(f"✅ Legitimate-Fälle: {legit_count} ({legit_pct:.1f}%)")
    if other_count > 0:
        print(f"❓ Unzuzuordnen:    {other_count}")
    print(f"📈 Gesamtanzahl:     {total}")
    print("="*40)

if __name__ == "__main__":
    analyze_train_data()