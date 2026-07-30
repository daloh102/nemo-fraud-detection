import sys
import json
import random
from pathlib import Path

# --------------------------------------------------------------------------
# 1. PFAD- & ORDNER-KONFIGURATION
# --------------------------------------------------------------------------
CURRENT_FILE_PATH = Path(__file__).resolve()

# Ermittelt das Projekt-Hauptverzeichnis (/nemo-fraud-detection)
if "src" in CURRENT_FILE_PATH.parts:
    # Falls das Skript in src/curator/ liegt -> 2 Ebenen hoch
    BASE_DIR = CURRENT_FILE_PATH.parents[2] if CURRENT_FILE_PATH.parent.name == "curator" else CURRENT_FILE_PATH.parents[1]
else:
    # Fallback für Aufruf außerhalb von src/
    BASE_DIR = Path("/data/nemo-fraud-detection")

DATA_DIR = BASE_DIR / "data"

# QUELL-DATEI (Wird ausschließlich GELESEN, nicht verändert)
ORIGINAL_RAW_PATH = DATA_DIR / "raw" / "dialogues_transcripts.jsonl"

# ZIEL-DATEI (Neue Datei für den NeMo Curator Testlauf)
NOISY_RAW_PATH = DATA_DIR / "raw" / "dialogues_transcripts_noisy.jsonl"

# Zielordner sicherstellen
NOISY_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 2. SAMPLE-DATEN FÜR NOISE-INJEKTION
# --------------------------------------------------------------------------
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

def load_jsonl(file_path):
    """Lädt JSONL-Datei zeilenweise."""
    records = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records

def format_to_text(item):
    """Extrahiert den Fließtext aus dem Record-Objekt."""
    if "text" in item and isinstance(item["text"], str):
        return item["text"]
    
    raw_response = item.get("response", "")
    try:
        parsed = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        if isinstance(parsed, dict) and "transcript" in parsed:
            return "\n".join([f"{t['speaker']}: {t['text']}" for t in parsed["transcript"]])
    except Exception:
        pass
        
    return ""

# --------------------------------------------------------------------------
# 3. DATEN EINLESEN (Ausschließlich aus den Original-Transkripten)
# --------------------------------------------------------------------------
print(f"📖 Lade Original-Transkripte (NUR LESEND) von: {ORIGINAL_RAW_PATH}")
input_records = load_jsonl(ORIGINAL_RAW_PATH)

if not input_records:
    print(f"❌ FEHLER: Keine Daten in '{ORIGINAL_RAW_PATH}' gefunden oder Datei existiert nicht! Abbruch.")
    sys.exit(1)

# --------------------------------------------------------------------------
# 4. KONTAMINIERUNG & NOISE-INJEKTION
# --------------------------------------------------------------------------
print(f"🧬 Injiziere Datenmüll, PII, Duplikate & Encoding-Fehler in {len(input_records)} Einträge...")
noisy_dataset = []
stats = {"duplicate": 0, "pii": 0, "garbage": 0, "encoding": 0, "clean": 0}

for idx, item in enumerate(input_records):
    call_id = item.get("call_id", item.get("id", f"CALL_{idx:04d}"))
    flat_text = format_to_text(item)

    if not flat_text:
        continue

    dice = random.random()
    
    if dice < 0.08:   # Exaktes Duplikat
        noisy_dataset.append({"call_id": call_id, "text": flat_text, "is_clean": True, "noise_type": "none"})
        noisy_dataset.append({"call_id": f"{call_id}_DUP", "text": flat_text, "is_clean": False, "noise_type": "exact_duplicate"})
        stats["duplicate"] += 1
        
    elif dice < 0.15: # PII-Daten
        pii_text = f"{flat_text}\nAnrufer: {random.choice(PII_SAMPLES)}"
        noisy_dataset.append({"call_id": call_id, "text": pii_text, "is_clean": False, "noise_type": "pii_injection"})
        stats["pii"] += 1
        
    elif dice < 0.20: # Müll-Text / Zu kurz
        noisy_dataset.append({"call_id": call_id, "text": random.choice(GARBAGE_SNIPPETS), "is_clean": False, "noise_type": "garbage_text"})
        stats["garbage"] += 1
        
    elif dice < 0.23: # Kaputtes UTF-8 Encoding (Mojibake)
        broken_text = flat_text.replace("ä", "Ã¤").replace("ö", "Ã¶").replace("ü", "Ã¼").replace("ß", "Ã\x9f")
        noisy_dataset.append({"call_id": call_id, "text": broken_text, "is_clean": False, "noise_type": "encoding_error"})
        stats["encoding"] += 1
        
    else: # Sauber belassen
        noisy_dataset.append({"call_id": call_id, "text": flat_text, "is_clean": True, "noise_type": "none"})
        stats["clean"] += 1

# --------------------------------------------------------------------------
# 5. EXPORT IN DIE NEUE TEST-DATEI
# --------------------------------------------------------------------------
print(f"💾 Schreibe verunreinigte Testdaten nach: {NOISY_RAW_PATH}")
with open(NOISY_RAW_PATH, "w", encoding="utf-8") as f:
    for rec in noisy_dataset:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("\n📊 INJEKTIONS-STATISTIK:")
print(f"  • Base Directory:                   {BASE_DIR}")
print(f"  • Original-Transkripte eingelesen: {ORIGINAL_RAW_PATH.name} ({len(input_records)} Zeilen)")
print(f"  • Duplikate injiziert:              {stats['duplicate']}")
print(f"  • PII-Einträge injiziert:           {stats['pii']}")
print(f"  • Müll-Texte injiziert:             {stats['garbage']}")
print(f"  • Encoding-Fehler erzeugt:          {stats['encoding']}")
print(f"  • Saubere Texte belassen:           {stats['clean']}")
print(f"\n✅ Erfolgreich ausgeführt! {len(noisy_dataset)} Einträge in '{NOISY_RAW_PATH.name}' erstellt.")