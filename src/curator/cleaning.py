import os
import re
import json
from nemo_curator.datasets import DocumentDataset
from nemo_curator import ScoreFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_RAW = os.path.join(BASE_DIR, "data", "raw", "call_transcripts_raw.jsonl")
INTERIM_CLEANED = os.path.join(BASE_DIR, "data", "interim", "cleaned_calls.jsonl")
INTERIM_PII_REMOVED = os.path.join(BASE_DIR, "data", "interim", "pii_removed_calls.jsonl")

def mask_pii_text(text):
    if not isinstance(text, str):
        return ""
    # Umlaute reparieren
    text = text.replace("Ã¤", "ä").replace("Ã¶", "ö").replace("Ã¼", "ü").replace("ÃŸ", "ß")
    # IBANs maskieren
    text = re.sub(r'DE\d{2}\s?(\d{4}\s?){4}\d{2}', '[IBAN_MASKIERT]', text)
    # Kreditkarten maskieren
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[KREDITKARTE_MASKIERT]', text)
    # Telefonnummern maskieren
    text = re.sub(r'\+?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{4,9}', '[TELEFON_MASKIERT]', text)
    return text

def run_cleaning_pipeline():
    print(f"🧹 Starte NeMo Curator Cleaning aus: {INPUT_RAW}")
    os.makedirs(os.path.dirname(INTERIM_CLEANED), exist_ok=True)
    
    # Dataset laden
    dataset = DocumentDataset.read_json(INPUT_RAW)
    
    # 1. PII-Anonymisierung
    print("🛡️ Anonymisiere PII und repariere Encoding...")
    pii_dataset = dataset.map(lambda doc: {
        "call_id": doc.get("call_id", doc.get("id")), 
        "text": mask_pii_text(doc.get("text", ""))
    })
    
    # Zwischenspeichern PII
    pii_docs = pii_dataset.to_dict()
    with open(INTERIM_PII_REMOVED, "w", encoding="utf-8") as f:
        for doc in pii_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    # 2. Längen-Filter (Mindestens 50 Zeichen)
    print("✂️ Filter zu kurze Müll-Texte...")
    valid_filter = ScoreFilter(
        score_fn=lambda doc: len(doc["text"]),
        score_type=int,
        min_score=50
    )
    cleaned_dataset = valid_filter(pii_dataset)
    
    # Ergebnis in interim/ cleaned_calls.jsonl schreiben
    cleaned_docs = cleaned_dataset.to_dict()
    with open(INTERIM_CLEANED, "w", encoding="utf-8") as f:
        for doc in cleaned_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    print(f"✅ Cleaned Dataset exportiert nach: {INTERIM_CLEANED} ({len(cleaned_docs)} Dokumente)")

if __name__ == "__main__":
    run_cleaning_pipeline()