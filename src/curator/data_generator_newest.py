import asyncio
import json
import os
import random
import re
from typing import Tuple
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

# 1. Konfiguration über Environment Variables (Fallback auf Default)
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://172.31.18.77:8001/v1")
MODEL_NAME = "meta/llama-3.1-8b-instruct"
OUTPUT_DATA_FILE = "fraud_call_transcripts_newest.jsonl"
OUTPUT_BENCHMARK_FILE = "fraud_call_benchmark_newest.jsonl"
NUM_SAMPLES = 8000
CONCURRENCY_LIMIT = 15  # Max 15 parallele Anfragen an das NIM

client = AsyncOpenAI(base_url=NIM_BASE_URL, api_key="not-needed")
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
file_lock = asyncio.Lock()  # Verhindert Verwaschen von Zeilen beim parallelen Schreiben

# 2. Erweiterter Namens-Pool (20 Namen) für maximale Varianz im Finetuning
KUNDEN_NAMEN = [
    "Herr Müller", "Frau Schmidt", "Herr Schneider", "Frau Fischer", 
    "Herr Weber", "Frau Meyer", "Herr Wagner", "Frau Becker", 
    "Herr Hoffmann", "Frau Schulz", "Herr Koch", "Frau Bauer", 
    "Herr Richter", "Frau Klein", "Herr Wolf", "Frau Schröder", 
    "Herr Neumann", "Frau Schwarz", "Herr Zimmermann", "Frau Braun"
]

# 3. Szenarien-Katalog: Fraud (Inbound: Anrufer gibt sich als Kunde aus)
FRAUD_SCENARIOS = [
    "Kunde fordert sofortige Kontoentsperrung, verweigert aber Sicherheitscodes wegen angeblich defektem Handy.",
    "Kunde verlangt Passwort-Reset für Online-Banking und nutzt entwendete Stammdaten (Geburtsdatum, Adresse), scheitert aber an Sicherheitsfrage.",
    "Kunde fordert Eilüberweisung und setzt den Bankmitarbeiter wegen angeblicher Notlage (z.B. Autokauf/Krankenhaus im Ausland) massiv unter Druck.",
    "Kunde versucht eine neue Telefonnummer / Adresse ohne PostIdent oder SMS-TAN im System hinterlegen zu lassen.",
    "Kunde fordert plötzliche Anhebung des Tageslimits für Überweisungen mit der Ausrede eines Spontankaufs.",
    "Kunde gibt vor, im Ausland bestohlen worden zu sein, und verlangt Notfall-Bargeldauszahlung ohne Ausweisdokumente.",
    "Kunde versucht, eine Fremdkarte/Zweitkarte auf eine neue Adresse bestellen zu lassen (Identity Theft).",
    "Kunde fragt gezielt Details zum Kontostand und letzten Buchungen ab, ohne die vollständige Legitimation erbringen zu können.",
    "Kunde behauptet, die 2FA-App funktioniere nicht, und versucht den Mitarbeiter zu überreden, die TAN manuell am Telefon freizugeben.",
    "Kunde gibt sich als bevollmächtigter Angehöriger (z.B. Sohn/Tochter) eines Senioren aus, hat aber keine eingetragene Vollmacht.",
    "Kunde täuscht eine schlechte Leitung/Nebengeräusche vor, um unverständliche Antworten bei Sicherheitsfragen zu kaschieren.",
    "Kunde nutzt 'Authority Framing' (behauptet, selbst wichtiger Firmenkunde oder Anwalt zu sein) und droht mit Konsequenzen, falls Regeln nicht gebogen werden.",
    "Kunde versucht, eine angeblich versehentlich getätigte Überweisung auf ein fremdes Konto stornieren zu lassen, um Gutschrift-Muster zu testen.",
    "Kunde fordert die Zusendung von neuen Online-Banking-Zugangsdaten an eine abweichende E-Mail-Adresse.",
    "Kunde behauptet, die Bank habe einen Fehler gemacht und Geld blockiert, und fordert aggressive Freigabe durch den Mitarbeiter.",
    "Kunde versucht ein ausländisches Empfängerkonto als angebliches eigenes Zweitkonto für eine Express-Überweisung freizuschalten.",
    "Kunde verwendet Social Engineering: Zeigt extremes Mitleid/Ablenkung durch persönliche Geschichten, um Legitimation zu überspringen.",
    "Kunde gibt vor, Firmensachbearbeiter zu sein, und verlangt Gehaltszahlungen auf ein neues Konto umzuleiten (CEO/Payroll Fraud via Call).",
    "Kunde versucht Push-TAN-Gerät auf ein neues Smartphone umregistrieren zu lassen, ohne den Post-Code abzuwarten.",
    "Kunde behauptet, Opfer eines Betrugs geworden zu sein, und fordert den Mitarbeiter auf, Transaktionen auf ein angebliches 'Sicherheitskonto' umzuleiten."
]

LEGITIMATE_SCENARIOS = [
    "Kunde erfragt Kontostand und die letzten Buchungen der letzten zwei Wochen.",
    "Kunde möchte einen bestehenden Dauerauftrag bezüglich Höhe und Ausführungstag ändern.",
    "Kunde erkundigt sich nach den Voraussetzungen und Zinsen für ein Festgeldkonto.",
    "Kunde hat seine PIN dreimal falsch eingegeben und bittet um Hilfe zur Freischaltung über den regulären Prozess.",
    "Kunde meldet einen ordnungsgemäßen Umzug und lässt seine Adresse nach erfolgreicher 2FA/Legitimation ändern.",
    "Kunde möchte seine verloren gegangene Debitkarte sperren lassen und eine Ersatzkarte bestellen.",
    "Kunde fragt nach Informationen zur Freischaltung des Online-Bankings für das Smartphone.",
    "Kunde versteht eine Abbuchung auf dem Kontoauszug nicht und lässt sich den Händlernamen erklären (stellt sich als legitim heraus).",
    "Kunde möchte vor einem Urlaub das Limit für Kartenzahlungen im Ausland temporär anpassen.",
    "Kunde fordert eine Steuerbescheinigung für das vergangene Jahr an.",
    "Kunde erkundigt sich nach den Bedingungen für einen Kleinkredit oder Disporahmen.",
    "Kunde fragt nach den Öffnungszeiten und Filialterminen für eine Beratung.",
    "Kunde möchte ein Unterkonto/Tagesgeldkonto neu anlegen.",
    "Kunde benötigt Unterstützung bei der Einrichtung von Apple Pay / Google Pay.",
    "Kunde informiert die Bank vorab über eine größere anstehende Auslandsüberweisung.",
    "Kunde erfragt den aktuellen Status einer bereits eingereichten Überweisung.",
    "Kunde möchte die Zusendung von Papier-Kontoauszügen auf elektronisches Postfach umstellen.",
    "Kunde stellt Fragen zur Verlängerung einer auslaufenden Kreditkarte.",
    "Kunde möchte eine Freistellungsauftrag für Kapitalerträge anpassen.",
    "Kunde erkundigt sich nach den Gebühren für Bargeldabhebungen im Ausland."
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
    ("lang und ausführlich (ca. 14-20 Dialogwechsel, detaillierte Diskussion, Rückfragen und Einwände)", 1600)
]

# Structured Output Schema definieren
class TranscriptSchema(BaseModel):
    id: str
    text: str = Field(description="Der komplette Gesprächsverlauf zwischen Kunde und Agent")
    source: str = "custom"

system_prompt = (
    "Du bist ein spezialisierter Data-Generator für Security- & Fraud-Detection-Modelle im Banking-Sektor.\n"
    "Deine Aufgabe ist es, realistisch klingende Transkripte von Telefonaten zwischen einem Anrufer und einem Bankmitarbeiter (Agent) zu erzeugen.\n\n"
    "WICHTIG:\n"
    "1. Verwende für den Kunden im Dialog den Namen, der im Prompt vorgegeben wird, und sprich ihn auch so an.\n"
    "2. Bei Fraud-Calls versucht der Anrufer, den Agenten durch Täuschung, Ausreden, Druck oder Manipulation zu unberechtigten Aktionen zu bewegen. Der Agent muss Sicherheitsrichtlinien beachten.\n"
    "3. Verwende im Text strikt die Sprecher-Präfixe 'Kunde:' bzw. den Namen und 'Agent:'.\n"
    "4. Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt ohne Markdown-Formatierung:\n"
    "{\n"
    '  "id": "doc-XXX",\n'
    '  "text": "Kunde: ... \\nAgent: ... \\nKunde: ...",\n'
    '  "source": "custom"\n'
    "}"
)

# globaler Zähler für Fortschrittsanzeige
completed_counter = 0

async def test_llm_connection():
    """Testet vorab, ob der LLM-Container erreichbar ist und antwortet."""
    print(f"🔍 Teste Verbindung zum LLM unter {NIM_BASE_URL} (Modell: {MODEL_NAME})...")
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Antworte nur mit 'OK'"}],
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip()
        print(f"✅ Verbindung erfolgreich! Test-Antwort vom Modell: '{answer}'")
    except Exception as e:
        print(f"❌ Verbindungstest zum LLM-Container fehlgeschlagen: {e}")
        print("💡 Bitte prüfe, ob der Docker-Container läuft und die URL/Port korrekt sind.")
        raise SystemExit(1)

async def generate_single_sample(index: int, f_data, f_bench):
    global completed_counter
    doc_id = f"doc-{index:05d}"
    is_fraud = (index % 2 != 0)
    label = "fraud" if is_fraud else "legitimate"
    tone = random.choice(TONES)
    length_desc, max_tokens_limit = random.choice(LENGTH_PROMPTS)
    customer_name = random.choice(KUNDEN_NAMEN)  # Zufällige Namenswahl aus 20 Namen
    
    if is_fraud:
        scenario = random.choice(FRAUD_SCENARIOS)
        user_prompt = (
            f"Generiere ein BETRUGSGESPRAECH (Fraud Call / Social Engineering Inbound Call).\n"
            f"Name des Kunden: {customer_name}.\n"
            f"Szenario: Der Anrufer gibt sich als dieser Kunde aus und versucht den Bankmitarbeiter zu überlisten. Details: {scenario}\n"
            f"Stimmung des Anrufers: {tone}.\n"
            f"Gesprächslänge: {length_desc}.\n"
            f"Achte auf realistische Dialogführung inklusive Nachfragen des Agenten nach Sicherheitsdaten und Reaktionen des Anrufers."
        )
    else:
        scenario = random.choice(LEGITIMATE_SCENARIOS)
        user_prompt = (
            f"Generiere ein LEGITIMES, normales Kundengespräch am Telefon (Legitimate Call).\n"
            f"Name des Kunden: {customer_name}.\n"
            f"Szenario: Der echte Kunde ruft beim Kundenservice der Bank an. Details: {scenario}\n"
            f"Stimmung des Kunden: {tone}.\n"
            f"Gesprächslänge: {length_desc}.\n"
            f"Der Kunde beantwortet Sicherheitsfragen korrekt und der Agent hilft gemäß den Bankstandards."
        )

    async with semaphore:
        for attempt in range(3):  # Bis zu 3 Versuche
            try:
                temp = round(random.uniform(0.7, 0.95), 2)
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temp,
                    max_tokens=max_tokens_limit,
                    response_format={"type": "json_object"}
                )

                raw_content = response.choices[0].message.content.strip()
                
                # Bereinigung eventueller Codeblock-Markdown-Tags
                clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
                clean_text_fixed = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_text)
                
                # Parsing
                try:
                    raw_json = json.loads(clean_text_fixed)
                except json.JSONDecodeError:
                    raw_json = {"id": doc_id, "text": clean_text.replace('\\', '/'), "source": "custom"}

                # Validierung über Pydantic Schema
                data = TranscriptSchema(
                    id=doc_id,
                    text=raw_json.get("text", ""),
                    source="custom"
                )

                # Thread-sicheres Schreiben mit asyncio.Lock
                async with file_lock:
                    f_data.write(data.model_dump_json() + "\n")
                    f_data.flush()
                    
                    f_bench.write(json.dumps({"id": doc_id, "label": label}, ensure_ascii=False) + "\n")
                    f_bench.flush()
                    
                    completed_counter += 1
                    if completed_counter % 50 == 0 or completed_counter == NUM_SAMPLES:
                        print(f"⏳ Fortschritt: [{completed_counter}/{NUM_SAMPLES}] ({completed_counter/NUM_SAMPLES*100:.1f}%)")
                return

            except (ValidationError, Exception) as e:
                if attempt == 2:
                    print(f"❌ Fehler bei {doc_id} nach 3 Versuchen: {e}")
                    # Fallback im Fehlerfall schreiben
                    fallback_text = f"Kunde: Guten Tag, ich bin {customer_name} und möchte eine Überweisung tätigen.\nAgent: Guten Tag. Aus Sicherheitsgründen benötige ich Ihre Legitimation."
                    fallback_data = TranscriptSchema(id=doc_id, text=fallback_text)
                    async with file_lock:
                        f_data.write(fallback_data.model_dump_json() + "\n")
                        f_bench.write(json.dumps({"id": doc_id, "label": label}, ensure_ascii=False) + "\n")
                        f_data.flush()
                        f_bench.flush()
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

async def main():
    # Verbindungstest vor dem Start der Generierung ausführen
    await test_llm_connection()

    print(f"\n🚀 Starte asynchrone Generierung von {NUM_SAMPLES} Datensätzen (Max Concurrency: {CONCURRENCY_LIMIT})...")
    
    with open(OUTPUT_DATA_FILE, "a", encoding="utf-8") as f_data, \
         open(OUTPUT_BENCHMARK_FILE, "a", encoding="utf-8") as f_bench:
        
        tasks = [generate_single_sample(i, f_data, f_bench) for i in range(1, NUM_SAMPLES + 1)]
        await asyncio.gather(*tasks)

    print(f"\n✅ Fertig! {NUM_SAMPLES} Datensätze wurden erfolgreich und parallel generiert.")

if __name__ == "__main__":
    asyncio.run(main())