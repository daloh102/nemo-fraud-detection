import os
import re
import json

# Import des NeMo Curator SDKs
from nemo_curator import ScoreFilter
from nemo_curator.datasets import DocumentDataset

# ==============================================================================
# NEMO CURATOR PIPELINE DEFINITION
# ==============================================================================
INPUT_PATH = "data/raw/nemo_designer_transcripts.jsonl"
CURATED_OUT_PATH = "data/curated/curated_transcripts.jsonl"

os.makedirs("data/curated", exist_ok=True)

print("⚡ Lade Rohdaten in das NeMo Curator Dataset...")
dataset = DocumentDataset.read_json(INPUT_PATH)

# 1. PII-Anonymisierungs-Funktion (IBANs, Kreditkarten & sensible Daten)
def mask_pii(text):
    if not isinstance(text, str):
        return text
    # IBAN Maskierung
    text = re.sub(r'DE\d{2}\s?(\d{4}\s?){4}\d{2}', '[IBAN_MASKIERT]', text)
    # Kreditkarten Maskierung
    text = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[KREDITKARTE_MASKIERT]', text)
    # Entfernen von Fehler-Markern aus der Inferenz
    if "PARSING_ERROR" in text:
        return ""
    return text

print("🛡️ Führe PII-Maskierung durch...")
dataset = dataset.map(lambda doc: {"id": doc["id"], "text": mask_pii(doc["input"])})

# 2. Qualitäts- & Längenfilterung via NeMo Curator Filter
print("🧹 Filter Mülldaten und zu kurze/defekte Dialoge heraus...")
# Entfernt leere Dokumente oder Texte unter 50 Zeichen
valid_length_filter = ScoreFilter(
    score_fn=lambda doc: len(doc["text"]),
    score_type=int,
    min_score=50
)
dataset = valid_length_filter(dataset)

# 3. Deduplizierung (Entfernen von exakten/fast-identischen Duplikaten)
print("✂️ Führe Deduplizierung durch...")
seen_hashes = set()
unique_docs = []

for doc in dataset.to_dict():
    # Einfacher Hash-Check über den gereinigten Text
    text_hash = hash(doc["text"])
    if text_hash not in seen_hashes and len(doc["text"].strip()) > 0:
        seen_hashes.add(text_hash)
        unique_docs.append(doc)

# Export der bereinigten Daten
print(f"💾 Schreibe bereinigte Daten ({len(unique_docs)} valide Einträge)...")
with open(CURATED_OUT_PATH, "w", encoding="utf-8") as f:
    for doc in unique_docs:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

print(f"\n✅ NeMo Curator Pipeline abgeschlossen! Datensatz liegt unter: {CURATED_OUT_PATH}")