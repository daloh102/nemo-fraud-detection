import sys
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any

# Native NeMo Curator Imports
from nemo_curator import Modify, ScoreFilter
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.modules import AddId
from nemo_curator.utils.distributed_utils import get_client

# ==============================================================================
# 1. PFAD- & ORDNER-KONFIGURATION
# ==============================================================================
BASE_DIR = Path("/data/nemo-fraud-detection")
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "raw" / "dialogues_transcripts_noisy.jsonl"
CURATED_OUT_PATH = DATA_DIR / "curated" / "dialogues_transcripts_curator.jsonl"
TEMP_EXPORT_DIR = DATA_DIR / "curated" / "_temp_curator_export"

CURATED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. STRIKTES ERROR HANDLING
# ==============================================================================
def validate_input_file(file_path: Path) -> int:
    print(f"🔍 Validiere Eingabedatei & Schema: {file_path.name}...")
    if not file_path.exists():
        print(f"❌ KRITISCHER FEHLER: Eingabedatei '{file_path}' existiert nicht!")
        sys.exit(1)

    total_lines = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            
            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"❌ SCHEMAFEHLER [Zeile {line_no}]: Kein valides JSON. Fehler: {e}")
                sys.exit(1)
            
            if "text" not in record:
                print(f"❌ SCHEMAFEHLER [Zeile {line_no}]: Pflichtfeld 'text' fehlt!")
                sys.exit(1)
                
            total_lines += 1

    if total_lines == 0:
        print("❌ KRITISCHER FEHLER: Eingabedatei ist vollkommen leer!")
        sys.exit(1)

    print(f"✅ Validation erfolgreich! {total_lines} Einträge für NeMo Curator bereit.\n")
    return total_lines

# ==============================================================================
# 3. CURATOR EXTENSIONS
# ==============================================================================
class FraudPiiAndEncodingModifier:
    def modify_document(self, doc: str) -> str:
        if not isinstance(doc, str):
            return doc
        
        # 1. UTF-8 Reparieren
        doc = doc.replace("Ã¤", "ä").replace("Ã¶", "ö").replace("Ã¼", "ü").replace("ÃŸ", "ß")
        
        # 2. PII Maskierungen
        doc = re.sub(r'DE\d{2}\s?(\d{4}\s?){4}\d{2}', '[IBAN_MASKIERT]', doc)
        doc = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[KREDITKARTE_MASKIERT]', doc)
        doc = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_MASKIERT]', doc)
        
        return doc


class MinLengthFilter(DocumentFilter):
    def __init__(self, min_length: int = 50, text_field: str = "text"):
        super().__init__()
        self.min_length = min_length
        self.text_field = text_field

    def score_document(self, doc: Any) -> int:
        if isinstance(doc, dict):
            text = doc.get(self.text_field, "")
        elif hasattr(doc, self.text_field):
            text = getattr(doc, self.text_field, "")
        elif isinstance(doc, str):
            text = doc
        else:
            return 0
        
        return len(str(text).strip())

    def keep_document(self, score: int) -> bool:
        return score >= self.min_length

# ==============================================================================
# 4. MAIN CURATOR PIPELINE
# ==============================================================================
def main():
    initial_count = validate_input_file(INPUT_PATH)

    print("🚀 Initialisiere NeMo Curator Execution Client...")
    client = get_client()

    print("⚡ 1. Lade Dataset in NeMo Curator...")
    dataset = DocumentDataset.read_json(str(INPUT_PATH))

    print("🆔 2. Generiere IDs für NeMo Curator...")
    add_id = AddId(id_field="id")
    dataset = add_id(dataset)

    print("🛡️ 3. Wende NeMo Curator Modifier an (PII & Encoding)...")
    cleaner = Modify(FraudPiiAndEncodingModifier())
    dataset = cleaner(dataset)

    print("🧹 4. Wende NeMo Curator MinLengthFilter an (Min 50 Zeichen)...")
    length_filter = ScoreFilter(
        MinLengthFilter(min_length=50, text_field="text"),
        score_type=int
    )
    dataset = length_filter(dataset)

    print("✂️ 5. Wende Exact-Deduplication im Dask-Dataset an...")
    dataset.df = dataset.df.drop_duplicates(subset=["text"])

    # Aufräumen alter Temp-Ordner und alter Ziel-Dateien
    if TEMP_EXPORT_DIR.exists():
        shutil.rmtree(TEMP_EXPORT_DIR)
    if CURATED_OUT_PATH.exists():
        if CURATED_OUT_PATH.is_dir():
            shutil.rmtree(CURATED_OUT_PATH)
        else:
            CURATED_OUT_PATH.unlink()

    print(f"💾 Schreibe kurierten Datensatz nach: {CURATED_OUT_PATH}")
    # Export in ein temporäres Verzeichnis
    dataset.to_json(str(TEMP_EXPORT_DIR), write_to_filename=False)

    # Zusammenführen/Verschieben der generierten Partitions-Datei in die finale Einzeldatei
    exported_files = list(TEMP_EXPORT_DIR.glob("*.json*")) + list(TEMP_EXPORT_DIR.glob("*.part"))
    if exported_files:
        # Falls Dask eine einzelne Partition erzeugt hat, einfach verschieben
        shutil.move(str(exported_files[0]), str(CURATED_OUT_PATH))
        shutil.rmtree(TEMP_EXPORT_DIR)
    else:
        # Fallback falls mehrere Partitions entstanden sind: Zusammenfügen
        with open(CURATED_OUT_PATH, "w", encoding="utf-8") as outfile:
            for part in sorted(TEMP_EXPORT_DIR.iterdir()):
                if part.is_file():
                    with open(part, "r", encoding="utf-8") as infile:
                        shutil.copyfileobj(infile, outfile)
        shutil.rmtree(TEMP_EXPORT_DIR)

    # Finale Auswertung
    final_docs = []
    with open(CURATED_OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_docs.append(json.loads(line))
    
    final_count = len(final_docs)
    total_removed = initial_count - final_count

    print("\n" + "="*80)
    print("📊 NEMO CURATOR RUN STATISTIK")
    print("="*80)
    print(f"  • Eingelesene Roh-Datensätze:      {initial_count}")
    print(f"  • Gefilterte / Bereinigte Einträge: {total_removed}")
    print("--------------------------------------------------------------------------------")
    print(f"  • Verbliebene kurierte Datensätze: {final_count}")
    print(f"  • Verwerfungsquote Total:          {(total_removed / initial_count):.2%}")
    print("="*80)
    print(f"\n✅ Pipeline erfolgreich beendet! Saubere Datei: {CURATED_OUT_PATH.name}")

if __name__ == "__main__":
    main()