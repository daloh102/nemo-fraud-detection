"""
Synthetischer Daten-Generator für Banking-Betrugserkennung (Fraud Detection SFT mit Guardrails)

Dieses Skript generiert asynchron realistische Gesprächstranskripte von Telefonaten 
zwischen Kunden und Bankmitarbeitern mithilfe von NeMo Curator, NVIDIA NIM APIs und NeMo Guardrails.

Autor:         Daniel Lohmann
Datum:         2026
"""

import asyncio
import json
import os
import random
import re
from typing import Tuple
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from nemo_curator import OpenAIClient
from nemo_curator.synthetic import NemotronGenerator
from nemoguardrails import LLMRails, RailsConfig
import wandb

# 1. Konfiguration über Environment Variables (Lokaler NIM Server Port 8800)
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://172.17.0.1:8800/v1")
GEN_MODEL = "meta/llama-3.1-8b-instruct"
JUDGE_MODEL = "meta/llama-3.1-8b-instruct"  # LLM-as-a-Judge Modell
OUTPUT_DATA_FILE = "fraud_call_transcripts_curator.jsonl"
OUTPUT_BENCHMARK_FILE = "fraud_call_benchmark_curator.jsonl"
NUM_SAMPLES = 100
CONCURRENCY_LIMIT = 8

# NeMo Curator Client Setup (angepasst auf lokalen OpenAI-kompatiblen NIM Server)
base_openai_client = OpenAI(base_url=NIM_BASE_URL, api_key="not-needed")
curator_openai_client = OpenAIClient(base_openai_client)
generator = NemotronGenerator(curator_openai_client)

semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
file_lock = asyncio.Lock()

# 2. Erweiterter Namens-Pool (20 Namen) für maximale Varianz im Finetuning
KUNDEN_NAMEN = [
    "Herr Müller", "Frau Schmidt", "Herr Schneider", "Frau Fischer", 
    "Herr Weber", "Frau Meyer", "Herr Wagner", "Frau Becker", 
    "Herr Hoffmann", "Frau Schulz", "Herr Koch", "Frau Bauer", 
    "Herr Richter", "Frau Klein", "Herr Wolf", "Frau Schröder", 
    "Herr Neumann", "Frau Schwarz", "Herr Zimmermann", "Frau Braun"
]

# 3. Szenarien-Katalog: Fraud & Legitimate
FRAUD_SCENARIOS = [
    "Kunde fordert sofortige Kontoentsperrung, verweigert aber Sicherheitscodes wegen angeblich defektem Handy.",
    "Kunde verlangt Passwort-Reset für Online-Banking und nutzt entwendete Stammdaten, scheitert aber an Sicherheitsfrage.",
    "Kunde fordert Eilüberweisung und setzt den Bankmitarbeiter wegen angeblicher Notlage massiv unter Druck.",
    "Kunde versucht eine neue Telefonnummer / Adresse ohne PostIdent oder SMS-TAN im System hinterlegen zu lassen.",
    "Kunde fordert plötzliche Anhebung des Tageslimits für Überweisungen mit der Ausrede eines Spontankaufs.",
    "Kunde gibt vor, im Ausland bestohlen worden zu sein, und verlangt Notfall-Bargeldauszahlung ohne Ausweisdokumente.",
    "Kunde versucht, eine Fremdkarte/Zweitkarte auf eine neue Adresse bestellen zu lassen (Identity Theft).",
    "Kunde fragt gezielt Details zum Kontostand und letzten Buchungen ab, ohne die vollständige Legitimation erbringen zu können.",
    "Kunde behauptet, die 2FA-App funktioniere nicht, und versucht den Mitarbeiter zu überreden, die TAN manuell freizugeben.",
    "Kunde gibt sich als bevollmächtigter Angehöriger eines Senioren aus, hat aber keine eingetragene Vollmacht."
]

LEGITIMATE_SCENARIOS = [
    "Kunde erfragt Kontostand und die letzten Buchungen der letzten zwei Wochen.",
    "Kunde möchte einen bestehenden Dauerauftrag bezüglich Höhe und Ausführungstag ändern.",
    "Kunde erkundigt sich nach den Voraussetzungen und Zinsen für ein Festgeldkonto.",
    "Kunde hat seine PIN dreimal falsch eingegeben und bittet um Hilfe zur Freischaltung über den regulären Prozess.",
    "Kunde meldet einen ordnungsgemäßen Umzug und lässt seine Adresse nach erfolgreicher 2FA/Legitimation ändern.",
    "Kunde möchte seine verloren gegangene Debitkarte sperren lassen und eine Ersatzkarte bestellen.",
    "Kunde fragt nach Informationen zur Freischaltung des Online-Bankings für das Smartphone.",
    "Kunde versteht eine Abbuchung auf dem Kontoauszug nicht und lässt sich den Händlernamen erklären.",
    "Kunde möchte vor einem Urlaub das Limit für Kartenzahlungen im Ausland temporär anpassen.",
    "Kunde fordert eine Steuerbescheinigung für das vergangene Jahr an."
]

TONES = [
    "sehr drängend, hektisch und autoritär",
    "verwirrt, unsicher und zögerlich",
    "extrem freundlich, charmant und ablenkend",
    "panisch, emotional aufgeladen und wütend",
    "sachlich, professionell und bestimmt",
    "ungeduldig und leicht genervt"
]

LENGTH_PROMPTS: list[Tuple[str, int]] = [
    ("kurz (ca. 4-6 Dialogwechsel, sehr direktes Gespräch)", 600),
    ("mittellang (ca. 8-12 Dialogwechsel, normale Gesprächslänge)", 1000),
    ("lang und ausführlich (ca. 14-20 Dialogwechsel, detaillierte Diskussion)", 1600)
]

# Structured Output Schema definieren
class TranscriptSchema(BaseModel):
    id: str
    text: str = Field(description="Der komplette Gesprächsverlauf zwischen Kunde und Agent")
    source: str = "nemo-curator-custom"

system_prompt = (
    "Du bist ein spezialisierter Data-Generator für Security- & Fraud-Detection-Modelle im Banking-Sektor.\n"
    "Deine Aufgabe ist es, realistisch klingende Transkripte von Telefonaten zwischen einem Anrufer und einem Bankmitarbeiter (Agent) zu erzeugen.\n\n"
    "WICHTIG:\n"
    "1. Verwende für den Kunden im Dialog den Namen, der im Prompt vorgegeben wird, und sprich ihn auch so an.\n"
    "2. Bei Fraud-Calls versucht der Anrufer, den Agenten durch Täuschung, Ausreden, Druck oder Manipulation zu unberechtigten Aktionen zu bewegen.\n"
    "3. Verwende im Text strikt die Sprecher-Präfixe 'Kunde:' bzw. den Namen und 'Agent:'.\n"
    "4. Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt ohne Markdown-Formatierung:\n"
    "{\n"
    '  "id": "doc-XXX",\n'
    '  "text": "Kunde: ... \\nAgent: ... \\nKunde: ...",\n'
    '  "source": "nemo-curator-custom"\n'
    "}"
)

completed_counter = 0

async def test_llm_connection():
    """Testet vorab über den NeMo Curator Client, ob der LLM-Container erreichbar ist."""
    print(f"🔍 Teste Verbindung zum LLM über NeMo Curator unter {NIM_BASE_URL} (Modell: {GEN_MODEL})...")
    try:
        response = curator_openai_client.query_model(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": "Antworte nur mit 'OK'"}],
            max_tokens=10
        )
        answer = response[0].strip()
        print(f"✅ Verbindung erfolgreich! Test-Antwort vom Modell: '{answer}'")
    except Exception as e:
        print(f"❌ Verbindungstest zum LLM-Container fehlgeschlagen: {e}")
        raise SystemExit(1)

async def evaluate_sample_quality(text: str) -> int:
    """LLM-as-a-Judge: Bewertet das generierte Transkript mit dem LLM."""
    eval_prompt = (
        "Bewerte dieses Bank-Transkript auf einer Skala von 1-5 hinsichtlich Realismus "
        "und Eignung für ein Fraud-Detection-Training (SFT).\n"
        "1 = Müll/unrealistisch, 5 = Perfekt.\n"
        f"Transkript: {text}\n"
        "Antworte NUR mit einem JSON: {\"score\": int}"
    )
    
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: curator_openai_client.query_model(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=50
            )
        )
        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", response[0].strip())
        data = json.loads(clean_json)
        return int(data.get("score", 0))
    except Exception:
        return 0

async def generate_single_sample(index: int, f_data, f_bench, rails_app: LLMRails):
    global completed_counter
    doc_id = f"doc-{index:05d}"
    is_fraud = (index % 2 != 0)
    label = "fraud" if is_fraud else "legitimate"
    tone = random.choice(TONES)
    length_desc, max_tokens_limit = random.choice(LENGTH_PROMPTS)
    customer_name = random.choice(KUNDEN_NAMEN)
    
    if is_fraud:
        scenario = random.choice(FRAUD_SCENARIOS)
        user_prompt = (
            f"Generiere ein BETRUGSGESPRAECH (Fraud Call / Social Engineering Inbound Call).\n"
            f"Name des Kunden: {customer_name}.\n"
            f"Szenario: Der Anrufer gibt sich als dieser Kunde aus und versucht den Bankmitarbeiter zu überlisten. Details: {scenario}\n"
            f"Stimmung des Anrufers: {tone}.\n"
            f"Gesprächslänge: {length_desc}."
        )
    else:
        scenario = random.choice(LEGITIMATE_SCENARIOS)
        user_prompt = (
            f"Generiere ein LEGITIMES, normales Kundengespräch am Telefon (Legitimate Call).\n"
            f"Name des Kunden: {customer_name}.\n"
            f"Szenario: Der echte Kunde ruft beim Kundenservice der Bank an. Details: {scenario}\n"
            f"Stimmung des Kunden: {tone}.\n"
            f"Gesprächslänge: {length_desc}."
        )

    async with semaphore:
        for attempt in range(3):
            try:
                temp = round(random.uniform(0.7, 0.95), 2)
                
                # Generierung über NeMo Guardrails (mit lokalem NIM-Modell)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Guardrails-Aufruf (asynchron)
                rails_response = await rails_app.generate_async(messages=messages)
                
                # Extraktion der Antwort je nach Rückgabeformat von LLMRails
                if isinstance(rails_response, dict):
                    raw_content = rails_response.get("content", str(rails_response))
                else:
                    raw_content = str(rails_response)

                clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
                clean_text_fixed = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_text)
                
                try:
                    raw_json = json.loads(clean_text_fixed)
                except json.JSONDecodeError:
                    raw_json = {"id": doc_id, "text": clean_text.replace('\\', '/'), "source": "nemo-curator-custom"}

                # Validierung über Pydantic Schema
                data = TranscriptSchema(
                    id=doc_id,
                    text=raw_json.get("text", raw_content),
                    source="nemo-curator-custom"
                )

                # LLM-as-a-Judge Qualitätsprüfung
                score = await evaluate_sample_quality(data.text)

                if score >= 4:  # Qualitäts-Gate
                    # Thread-sicheres Schreiben mit asyncio.Lock
                    async with file_lock:
                        f_data.write(data.model_dump_json() + "\n")
                        f_data.flush()
                        
                        f_bench.write(json.dumps({"id": doc_id, "label": label, "score": score}, ensure_ascii=False) + "\n")
                        f_bench.flush()
                        
                        # wandb Logging für jeden validen Sample
                        wandb.log({
                            "sample_score": score,
                            "is_compliant": 1 if not is_fraud else 0,
                            "is_accepted": 1
                        })

                        completed_counter += 1
                        if completed_counter % 50 == 0 or completed_counter == NUM_SAMPLES:
                            print(f"⏳ Fortschritt (Curated & Judged): [{completed_counter}/{NUM_SAMPLES}] ({completed_counter/NUM_SAMPLES*100:.1f}%)")
                    return
                else:
                    if attempt == 2:
                        wandb.log({"is_accepted": 0})
                        return

            except Exception as e:
                if attempt == 2:
                    print(f"❌ Guardrails-Fehler bei {doc_id} nach 3 Versuchen: {e}")
                await asyncio.sleep(1 * (attempt + 1))

async def main():
    # wandb initialisieren
    wandb.init(project="nemo-fraud-detection-curator", name="synthetic-generation-with-guardrails")

    await test_llm_connection()

    # NeMo Guardrails Konfiguration für den lokalen NIM-Server laden
    config = RailsConfig.from_content(
        colang_content="",
        yaml_content=f"""
models:
  - type: main
    engine: openai
    model: {GEN_MODEL}
    parameters:
      base_url: {NIM_BASE_URL}
      api_key: not-needed
        """
    )
    rails_app = LLMRails(config)

    print(f"\n🚀 Starte Generierung mit Guardrails & Curator von bis zu {NUM_SAMPLES} Datensätzen (Max Concurrency: {CONCURRENCY_LIMIT})...")
    
    with open(OUTPUT_DATA_FILE, "a", encoding="utf-8") as f_data, \
         open(OUTPUT_BENCHMARK_FILE, "a", encoding="utf-8") as f_bench:
        
        tasks = [generate_single_sample(i, f_data, f_bench, rails_app) for i in range(1, NUM_SAMPLES + 1)]
        await asyncio.gather(*tasks)

    print(f"\n✅ Fertig! Validierte Datensätze wurden in {OUTPUT_DATA_FILE} und {OUTPUT_BENCHMARK_FILE} gespeichert.")

    # Datensatz als wandb Artifact hochladen
    print("📦 Lade Datensatz als wandb Artifact hoch...")
    artifact = wandb.Artifact(name="fraud-transcripts-dataset", type="dataset")
    artifact.add_file(OUTPUT_DATA_FILE)
    artifact.add_file(OUTPUT_BENCHMARK_FILE)
    wandb.log_artifact(artifact)
    print("✨ wandb Artifact erfolgreich hochgeladen!")

    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())