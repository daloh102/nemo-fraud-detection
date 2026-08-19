"""
Modul: NeMo Fraud Detection - Noise Injection Pipeline
======================================================

Beschreibung:
Dieses Skript bereitet Trainings- oder Testdaten für das Projekt 'nemo-fraud-detection' 
vor. Es liest rohe Dialog-Transkripte im JSONL-Format ein und injiziert gezielt künstliches 
Rauschen, um die Robustheit nachfolgender NLP- und Machine-Learning-Modelle zu testen.

Simulierte Fehlertypen & Injektionswahrscheinlichkeiten:
  • Exakte Duplikate         (Standard: 8%)
  • PII-Lecks (Personenbez.) (Standard: 7%)
  • Textmüll / Garbage       (Standard: 5%)
  • Encoding-Fehler (Umlaute) (Standard: 3%)
  • Saubere Originaltexte    (~77%)

Eingabe:
  - /data/nemo-fraud-detection/transcripts.jsonl

Ausgabe:
  - /data/nemo-fraud-detection/transcripts_noisy.jsonl
  - Konsolenausgabe mit detaillierter Injektions-Statistik.

Verwendung:
  python <skript_name>.py

Autor:         Daniel Lohmann
Datum:         2026
Erfolgreich geprüft am: 19.08.2026
"""

import sys
import json
import random
from pathlib import Path
from typing import Any, Dict, List

# ==============================================================================
# 1. KONFIGURATION & PFADE
# ==============================================================================
BASE_DIR = Path("/data")
DATA_DIR = BASE_DIR / "nemo-fraud-detection" / "data" / "raw"

ORIGINAL_RAW_PATH = DATA_DIR / "transcripts.jsonl"
NOISY_RAW_PATH = DATA_DIR / "transcripts_noisy.jsonl"

NOISY_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

# Wahrscheinlichkeiten für die Noise-Injektion (anpassbar)
PROB_DUPLICATE = 0.08
PROB_PII = 0.07          # Kumuliert bis 0.15
PROB_GARBAGE = 0.05      # Kumuliert bis 0.20
PROB_ENCODING = 0.03     # Kumuliert bis 0.23
# Rest (~77%) bleibt sauber

PII_SAMPLES = [
    "Meine Kreditkartennummer lautet 4532-8921-1029-4411 mit CVV 892.",
    "Sie erreichen mich unter max.mustermann@example.de oder Mobil: +49 171 1234567.",
    "Meine IBAN lautet DE89 3704 0044 0532 0130 00, Inhaber ist Thomas Müller.",
    "Ich wohne in der Hauptstraße 45, 10115 Berlin. Geburtsdatum ist der 14.05.1982.",
    "Meine Sozialversicherungsnummer ist 12 140582 M 043."
]

GARBAGE_SNIPPETS = [
    "???",
    "N/A",
    "asdfghjkl;",
    "NULL",
    "[ERROR_VOICEMAIL_RECORDING_CORRUPTED]",
    "CLICK... BEEP... BEEP...",
    "a"
]

# ==============================================================================
# 2. HILFSFUNKTIONEN
# ==============================================================================
def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Lädt eine JSONL-Datei zeilenweise und fängt Encoding- oder Parsing-Fehler ab."""
    records = []
    if not file_path.exists():
        return records
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                print(f"⚠️ Warnung: Ungültiges JSON in Zeile {line_no}: {e}")
    return records

def format_to_text(item: Dict[str, Any]) -> str:
    """Extrahiert den Fließtext oder baut das Transkript sicher zusammen."""
    if "text" in item and isinstance(item["text"], str):
        return item["text"]
    
    raw_response = item.get("response", "")
    if not raw_response:
        return ""
        
    try:
        parsed = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        if isinstance(parsed, dict) and "transcript" in parsed:
            return "\n".join([f"{t.get('speaker', 'Unbekannt')}: {t.get('text', '')}" for t in parsed["transcript"]])
    except (json.JSONDecodeError, TypeError):
        pass
        
    return ""

# ==============================================================================
# 3. HAUPTLOGIK
# ==============================================================================
def main():
    print(f"📖 Lade Original-Transkripte von: {ORIGINAL_RAW_PATH}")
    input_records = load_jsonl(ORIGINAL_RAW_PATH)

    if not input_records:
        print(f"❌ FEHLER: Keine Daten in '{ORIGINAL_RAW_PATH}' gefunden oder Datei existiert nicht! Abbruch.")
        sys.exit(1)

    print(f"🧬 Injiziere Datenmüll, PII, Duplikate & Encoding-Fehler in {len(input_records)} Einträge...")
    noisy_dataset = []
    stats = {"duplicate": 0, "pii": 0, "garbage": 0, "encoding": 0, "clean": 0}

    for idx, item in enumerate(input_records):
        call_id = item.get("call_id", item.get("id", f"CALL_{idx:04d}"))
        flat_text = format_to_text(item)

        if not flat_text:
            continue

        dice = random.random()
        
        if dice < PROB_DUPLICATE:
            noisy_dataset.append({"call_id": call_id, "text": flat_text, "is_clean": True, "noise_type": "none"})
            noisy_dataset.append({"call_id": f"{call_id}_DUP", "text": flat_text, "is_clean": False, "noise_type": "exact_duplicate"})
            stats["duplicate"] += 1
            
        elif dice < (PROB_DUPLICATE + PROB_PII):
            pii_text = f"{flat_text}\nAnrufer: {random.choice(PII_SAMPLES)}"
            noisy_dataset.append({"call_id": call_id, "text": pii_text, "is_clean": False, "noise_type": "pii_injection"})
            stats["pii"] += 1
            
        elif dice < (PROB_DUPLICATE + PROB_PII + PROB_GARBAGE):
            noisy_dataset.append({"call_id": call_id, "text": random.choice(GARBAGE_SNIPPETS), "is_clean": False, "noise_type": "garbage_text"})
            stats["garbage"] += 1
            
        elif dice < (PROB_DUPLICATE + PROB_PII + PROB_GARBAGE + PROB_ENCODING):
            broken_text = flat_text.replace("ä", "Ã¤").replace("ö", "Ã¶").replace("ü", "Ã¼").replace("ß", "Ã\x9f")
            noisy_dataset.append({"call_id": call_id, "text": broken_text, "is_clean": False, "noise_type": "encoding_error"})
            stats["encoding"] += 1
            
        else:
            noisy_dataset.append({"call_id": call_id, "text": flat_text, "is_clean": True, "noise_type": "none"})
            stats["clean"] += 1

    print(f"💾 Schreibe verunreinigte Testdaten nach: {NOISY_RAW_PATH}")
    with open(NOISY_RAW_PATH, "w", encoding="utf-8") as f:
        for rec in noisy_dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n📊 INJEKTIONS-STATISTIK:")
    print(f"  • Base Directory:                 {BASE_DIR}")
    print(f"  • Original-Transkripte eingelesen: {ORIGINAL_RAW_PATH.name} ({len(input_records)} Zeilen)")
    print(f"  • Duplikate injiziert:            {stats['duplicate']}")
    print(f"  • PII-Einträge injiziert:         {stats['pii']}")
    print(f"  • Müll-Texte injiziert:           {stats['garbage']}")
    print(f"  • Encoding-Fehler erzeugt:        {stats['encoding']}")
    print(f"  • Saubere Texte belassen:         {stats['clean']}")
    print(f"\n✅ Erfolgreich ausgeführt! {len(noisy_dataset)} Einträge in '{NOISY_RAW_PATH.name}' erstellt.")

if __name__ == "__main__":
    main()