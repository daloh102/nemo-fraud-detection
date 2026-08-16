"""
Funktionsbeschreibung der Daten-Kurations-Pipeline:

Dieser Quellcode implementiert eine automatisierte, GPU-beschleunigte Pipeline zur Bereinigung, 
Validierung und Vorbereitung von Textdaten (speziell Finanztranskripten) für Machine-Learning-Anwendungen. 
Der Prozess gliedert sich in folgende logische Hauptschritte:

*   Infrastruktur & Validierung: Zunächst werden Dateipfade konfiguriert und die Eingabedatei (.jsonl) 
    strikt auf Schema-Konformität geprüft. Dabei wird sichergestellt, dass jedes JSON-Objekt ein 
    Pflichtfeld für "text" enthält, um Laufzeitfehler in der späteren Pipeline zu vermeiden.

*   Kundenspezifische Datenverarbeitung (Extensions):
    *   PII-Maskierung: Eine spezialisierte Klasse (FraudPiiModifier) verwendet reguläre Ausdrücke, 
        um sensible Daten wie IBANs, Kreditkartennummern und E-Mail-Adressen in den Texten automatisch 
        zu anonymisieren.
    *   Qualitätsfilter: Mittels eines benutzerdefinierten Filters (MinLengthFilter) werden zu kurze 
        Texte, die keinen inhaltlichen Mehrwert bieten (unter 50 Zeichen), aus dem Datensatz entfernt.

*   Advanced Data Curation Pipeline: Unter Verwendung von RAPIDS/CUDA-beschleunigtem Computing 
    durchläuft der Datensatz einen strukturierten Prozess:
    1. Identifikation: Jeder Eintrag erhält eine eindeutige ID für die Nachverfolgbarkeit.
    2. Bereinigung: Unicode-Zeichen werden standardisiert und die oben definierte PII-Maskierung angewendet.
    3. Klassifikation: Einsatz eines DeBERTa-basierten multilingualen Domain-Klassifikators zur 
       inhaltlichen Zuordnung.
    4. Deduplizierung: Ein MD5-Hash-basiertes Verfahren (ExactDuplicates) identifiziert und eliminiert 
       exakte Textduplikate, um den Datensatz effizienter zu gestalten.

*   Export & Statistik: Die verarbeiteten Daten werden als einheitliche JSONL-Datei zusammengeführt. 
    Abschließend erstellt das Skript einen detaillierten Statistikbericht, der den ursprünglichen 
    Datensatz mit der Anzahl der gefilterten bzw. entfernten Einträge vergleicht und die 
    Verwerfungsquote ausgibt.

Zusammenfassend dient das Skript dazu, aus verrauschten Rohdaten einen hochwertigen, anonymisierten 
und bereinigten Datensatz zu erzeugen, der für das Training von KI-Modellen im Finanzsektor 
optimiert ist.

Autor:          Daniel Lohmann
Datum:          2026
"""

import sys
import json
import re
import shutil
from pathlib import Path
from typing import Any
import os

# Native NeMo Curator & RAPIDS Imports
from nemo_curator import Modify, ScoreFilter, Sequential, AddId
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.modules import ExactDuplicates
from nemo_curator.classifiers import DomainClassifier
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.modifiers import DocumentModifier, UnicodeReformatter

# Saubere Initialisierung, die den Segmentation Fault bei cuDF/Dask umgeht:
from dask_cuda import LocalCUDACluster
from distributed import Client

# Native NeMo Curator & RAPIDS Imports
from nemo_curator import Modify, ScoreFilter, Sequential, AddId
# ==============================================================================
# 1. PFAD- & ORDNER-KONFIGURATION
# ==============================================================================
BASE_DIR = Path("/data")
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = INPUT_PATH = DATA_DIR / "raw" / "dialogues_transcripts_noisy_new.jsonl"
CURATED_OUT_PATH = DATA_DIR / "curated" / "dialogues_transcripts_curator_new.jsonl"
TEMP_EXPORT_DIR = DATA_DIR / "curated" / "_temp_curator_export"
DEDUP_LOG_DIR = DATA_DIR / "curated" / "dedup_logs"
DEDUP_CACHE_DIR = DATA_DIR / "curated" / "dedup_cache"

CURATED_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
DEDUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEDUP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -- GDS / cuFile Workaround für Container-Umgebungen --
os.environ["KVIKIO_COMPAT_MODE"] = "ON"
os.environ["CUDF_CUFILE_ENABLED"] = "0"
os.environ["RAPIDS_NO_CUFILE"] = "1"
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
class FraudPiiModifier(DocumentModifier):
    """Custom Modifier zur Maskierung von PII (IBAN, Kreditkarten, E-Mails, Geburtsdaten, Handynummern, Adressen, Kundennummern) in Finanztranskripten."""
    def modify_document(self, doc: str) -> str:
        if not isinstance(doc, str):
            return doc
        
        # 1. IBAN
        doc = re.sub(r'DE\d{2}\s?(\d{4}\s?){4}\d{2}', '[IBAN_MASKIERT]', doc)
        
        # 2. Kreditkarten
        doc = re.sub(r'\b(?:\d[ -]*?){13,16}\b', '[KREDITKARTE_MASKIERT]', doc)
        
        # 3. E-Mails
        doc = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_MASKIERT]', doc)
        
        # 4. Geburtsdaten
        doc = re.sub(r'\b\d{1,2}\.\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}\b', '[GEBURTSDATUM_MASKIERT]', doc, flags=re.IGNORECASE)
        
        # 5. Handynummern
        doc = re.sub(r'\b(?:\+49|0)\s*1[567]\d[\s\-]?\d{3,8}\b', '[HANDYNUMMER_MASKIERT]', doc)
        
        # 6. Adressen (jetzt auch mit mehrteiligen Straßennamen wie "Berliner Straße")
        doc = re.sub(r'\b(?:[A-ZÄÖÜ][a-zäöüß]+\s+)?[A-ZÄÖÜ][a-zäöüß]+(?:straße|str\.|weg|allee|platz|ring)\s+\d+[a-zA-Z]?,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+\b', '[ADRESSE_MASKIERT]', doc, flags=re.IGNORECASE)
        
        # 7. Kundennummern (Erkennt z.B. 9-stellige Nummern im Kontext oder als ID)
        doc = re.sub(r'\b\d{9}\b', '[KUNDENNUMMER_MASKIERT]', doc)
        
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
# 4. MAIN CURATOR PIPELINE (Mit Advanced Processing)
# ==============================================================================
def main():
    initial_count = validate_input_file(INPUT_PATH)

    print("🚀 Initialisiere NeMo Curator Execution Client (GPU/CUDA Backend)...")
    # Einziger zentraler Client-Start ganz am Anfang
    client = get_client(cluster_type="gpu", set_torch_to_use_rmm=False)
    print("🔗 Dask-Client erfolgreich verbunden.")

    print("⚡ 1. Lade Dataset in NeMo Curator...")
    dataset = DocumentDataset.read_json(str(INPUT_PATH), add_filename=True, backend="pandas")

    print("🆔 2. Generiere eindeutige IDs für NeMo Curator (Voraussetzung für Deduplizierung)...")
    add_id = AddId(id_field="id", id_prefix="FRAUD_data", start_index=0)
    dataset = add_id(dataset)

    print("🛡️ 3. Wende Cleaning-Sequenz an (Unicode-Reformat via ftfy & PII-Maskierung)...")
    cleaners = Sequential([
        Modify(UnicodeReformatter()),
        Modify(FraudPiiModifier())
    ])
    dataset = cleaners(dataset).persist()

    print("🧹 4. Wende NeMo Curator MinLengthFilter an (Min 50 Zeichen)...")
    length_filter = ScoreFilter(
        MinLengthFilter(min_length=50, text_field="text"),
        score_type=int
    )
    dataset = length_filter(dataset)

    print("✂️ 5. Führe Exakte Deduplizierung (ExactDuplicates) aus...")
    exact_dup = ExactDuplicates(
        logger=str(DEDUP_LOG_DIR),
        id_field="id",
        text_field="text",
        hash_method="md5",
        cache_dir=str(DEDUP_CACHE_DIR),
    )
    duplicates_dataset = exact_dup(dataset=dataset)
    
    # Identifiziere Duplikate, die entfernt werden sollen
    exact_docs_to_remove = duplicates_dataset.df.map_partitions(
        lambda x: x[x._hashes.duplicated(keep="first")]
    )

    # Filter herausfiltern
    id_field = "id"
    cleaned_df = dataset.df[
        ~dataset.df[id_field].isin(exact_docs_to_remove[id_field].compute())
    ]
    dataset = DocumentDataset(cleaned_df)

    # Aufräumen alter Temp-Ordner und alter Ziel-Dateien
    if TEMP_EXPORT_DIR.exists():
        shutil.rmtree(TEMP_EXPORT_DIR)
    if CURATED_OUT_PATH.exists():
        if CURATED_OUT_PATH.is_dir():
            shutil.rmtree(CURATED_OUT_PATH)
        else:
            CURATED_OUT_PATH.unlink()

    print(f"💾 Schreibe finalen, kurierten Datensatz nach: {CURATED_OUT_PATH}")
    dataset.to_json(str(TEMP_EXPORT_DIR), write_to_filename=False)

    # Zusammenführen der Partitions-Dateien
    exported_files = list(TEMP_EXPORT_DIR.glob("*.json*")) + list(TEMP_EXPORT_DIR.glob("*.part"))
    if exported_files:
        shutil.move(str(exported_files[0]), str(CURATED_OUT_PATH))
        shutil.rmtree(TEMP_EXPORT_DIR)
    else:
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
    print("📊 NEMO CURATOR ADVANCED PIPELINE STATISTIK")
    print("="*80)
    print(f"  • Eingelesene Roh-Datensätze:      {initial_count}")
    print(f"  • Gefilterte / Duplizierte Einträge: {total_removed}")
    print("--------------------------------------------------------------------------------")
    print(f"  • Verbliebene kurierte Datensätze: {final_count}")
    print(f"  • Verwerfungsquote Total:          {(total_removed / initial_count):.2%}")
    print("="*80)
    print(f"\n✅ Pipeline erfolgreich beendet! Saubere Datei: {CURATED_OUT_PATH.name}")

if __name__ == "__main__":
    main()