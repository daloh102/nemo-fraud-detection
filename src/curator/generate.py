import asyncio
import json
import os
import random
import re
from typing import Tuple
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

# 1. Konfiguration über Environment Variables
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://172.31.18.77:8001/v1")
MODEL_NAME = "meta/llama-3.1-8b-instruct"
OUTPUT_DATA_FILE = "fraud_call_transcripts_new.jsonl"
OUTPUT_BENCHMARK_FILE = "fraud_call_benchmark_new.jsonl"
NUM_SAMPLES = 8000
CONCURRENCY_LIMIT = 15

client = AsyncOpenAI(base_url=NIM_BASE_URL, api_key="not-needed")
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
file_lock = asyncio.Lock()

KUNDEN_NAMEN = [
    "Herr Müller", "Frau Schmidt", "Herr Schneider", "Frau Fischer", 
    "Herr Weber", "Frau Meyer", "Herr Wagner", "Frau Becker", 
    "Herr Hoffmann", "Frau Schulz", "Herr Koch", "Frau Bauer", 
    "Herr Richter", "Frau Klein", "Herr Wolf", "Frau Schröder", 
    "Herr Neumann", "Frau Schwarz", "Herr Zimmermann", "Frau Braun"
]

FRAUD_SCENARIOS = [
    ("Kunde fordert sofortige Kontoentsperrung, verweigert aber Sicherheitscodes wegen angeblich defektem Handy.", "identity_theft", "medium"),
    ("Kunde verlangt Passwort-Reset für Online-Banking und nutzt entwendete Stammdaten.", "account_takeover", "hard"),
    ("Kunde fordert Eilüberweisung und setzt den Bankmitarbeiter wegen angeblicher medizinischer Notfälle im Ausland massiv unter Druck.", "urgency_fraud", "medium"),
    ("Kunde versucht eine neue Telefonnummer ohne PostIdent oder SMS-TAN im System hinterlegen zu lassen.", "sim_swap", "hard"),
    ("Kunde fordert plötzliche Anhebung des Tageslimits für Überweisungen mit der Ausrede eines Spontankaufs.", "account_takeover", "easy"),
    ("Kunde gibt vor, im Ausland bestohlen worden zu sein, und verlangt Notfall-Bargeldauszahlung ohne Ausweis.", "identity_theft", "medium"),
    ("Kunde versucht, eine Fremdkarte auf eine neue Adresse bestellen zu lassen (Identity Theft).", "identity_theft", "hard"),
    ("Kunde fragt gezielt Kontostand ab, ohne die vollständige Legitimation erbringen zu können.", "social_engineering", "easy"),
    ("Kunde behauptet, die 2FA-App funktioniere nicht, und versucht den Mitarbeiter zu überreden, die TAN manuell am Telefon freizugeben.", "sim_swap", "hard"),
    ("Kunde gibt sich als bevollmächtigter Angehöriger aus, hat aber keine eingetragene Vollmacht im System.", "identity_theft", "medium"),
    ("Kunde täuscht schlechte Verbindung vor, um unverständliche Antworten bei Sicherheitsfragen zu kaschieren.", "technical_excuse", "easy"),
    ("Kunde nutzt Authority Framing (behauptet, wichtiger Anwalt zu sein) und droht mit Konsequenzen.", "urgency_fraud", "hard"),
    ("Kunde versucht, eine angeblich versehentlich getätigte Überweisung auf ein fremdes Konto stornieren zu lassen.", "social_engineering", "medium"),
    ("Kunde fordert Zusendung von neuen Zugangsdaten an eine abweichende E-Mail-Adresse.", "account_takeover", "medium"),
    ("Kunde behauptet, die Bank habe einen Fehler gemacht, und fordert aggressive Freigabe.", "urgency_fraud", "easy"),
    ("Kunde versucht ein ausländisches Empfängerkonto als eigenes Zweitkonto für Express-Überweisung freizuschalten.", "account_takeover", "hard"),
    ("Kunde verwendet Social Engineering mit emotionalen Geschichten, um Legitimation zu überspringen.", "emotional_manipulation", "medium"),
    ("Kunde gibt vor, Firmensachbearbeiter zu sein, und verlangt Gehaltsumleitung auf ein neues Konto.", "identity_theft", "hard"),
    ("Kunde versucht Push-TAN-Gerät auf ein neues Smartphone umzuordnen, ohne den Freischaltcode abzuwarten.", "sim_swap", "medium"),
    ("Kunde behauptet, Opfer von Betrug zu sein, und fordert Umleitung auf ein angebliches Sicherheitskonto.", "social_engineering", "hard")
]

LEGITIMATE_SCENARIOS = [
    ("Kunde erfragt Kontostand und die letzten Buchungen.", "legitimate", "easy"),
    ("Kunde möchte einen bestehenden Dauerauftrag ändern.", "legitimate", "easy"),
    ("Kunde erkundigt sich nach den Zinsen für ein Festgeldkonto.", "legitimate", "easy"),
    ("Kunde hat PIN falsch eingegeben und bittet um reguläre Freischaltung.", "legitimate", "medium"),
    ("Kunde meldet ordnungsgemäßen Umzug und ändert Adresse nach erfolgreicher 2FA.", "legitimate", "medium"),
    ("Kunde möchte verlorene Debitkarte sperren und Ersatzkarte bestellen.", "legitimate", "easy"),
    ("Kunde fragt nach Anleitung zur Freischaltung von Online-Banking auf dem Smartphone.", "legitimate", "easy"),
    ("Kunde versteht eine Abbuchung nicht und klärt den Händlernamen.", "legitimate", "easy"),
    ("Kunde möchte Limit für Kartenzahlungen im Ausland temporär anpassen.", "legitimate", "easy"),
    ("Kunde fordert Steuerbescheinigung an.", "legitimate", "easy"),
    ("Kunde erkundigt sich nach Bedingungen für Kleinkredit.", "legitimate", "medium"),
    ("Kunde fragt nach Öffnungszeiten der Filiale.", "legitimate", "easy"),
    ("Kunde möchte ein Tagesgeldkonto anlegen.", "legitimate", "easy"),
    ("Kunde benötigt Hilfe bei Apple Pay / Google Pay Einrichtung.", "legitimate", "medium"),
    ("Kunde informiert vorab über größere Auslandsüberweisung.", "legitimate", "medium"),
    ("Kunde erfragt Status einer eingereichten Überweisung.", "legitimate", "easy"),
    ("Kunde stellt Kontoauszüge aufs elektronische Postfach um.", "legitimate", "easy"),
    ("Kunde fragt nach Verlängerung der Kreditkarte.", "legitimate", "easy"),
    ("Kunde möchte Freistellungsauftrag anpassen.", "legitimate", "easy"),
    ("Kunde erkundigt sich nach Gebühren im Ausland.", "legitimate", "easy")
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
    ("kurz (ca. 4-6 Dialogwechsel)", 600),
    ("mittellang (ca. 8-12 Dialogwechsel)", 1000),
    ("lang und ausführlich (ca. 14-20 Dialogwechsel)", 1600)
]

system_prompt = """Du bist ein Generator synthetischer Trainingsdaten für Security- und Fraud-Detection-Modelle im Banking.

Deine Aufgabe besteht darin, realistische Transkripte von Telefongesprächen zwischen einem Bankkunden und einem Bankmitarbeiter (Agent) zu erzeugen.

ANFORDERUNGEN

1. Erzeuge ausschließlich fiktive Gespräche.

2. Der Kundenname wird im User-Prompt vorgegeben. Ersetze den Platzhalter <Name> im gesamten Dialog konsequent durch diesen echten Namen (z.B. 'Herr Müller:').

3. Verwende ausschließlich die Sprecherpräfixe:
<Echter Name>:
Agent:

4. SPRACHSTIL & NATURALISMUS:
- Die Dialoge müssen absolut authentisch und wie echte deutsche Telefonate klingen.
- Vermeide hölzerne, gestelzte oder roboterhafte Formulierungen (kein "Guten Tag, freundlicherweise" oder überschwängliches Theater).
- Der Agent spricht professionell, sachlich, aber menschlich.
- Der Kunde redet natürlich, mit Unterbrechungen, umgangssprachlichen Elementen oder typischen Satzanfängen.
- Ein Dialog muss sauber und logisch enden (kein "Auf Wiederhören... Guten Tag").

5. Bei Fraud-Szenarien versucht der Kunde den Agenten mittels Social Engineering zu manipulieren, z.B.
- Identitätsbetrug
- Kontoübernahme
- SIM-Swap
- Dringlichkeit
- emotionale Manipulation
- falsche Notfälle
- technische Ausreden
- Druck auf den Agenten

6. Der Agent handelt professionell und entsprechend üblicher Bankrichtlinien:
- Identität prüfen
- Sicherheitsfragen stellen
- kritische Aktionen verweigern
- Alternativen anbieten
- Verdachtsfälle eskalieren

7. Jeder Dialog soll sich deutlich von vorherigen unterscheiden.

8. Gib ausschließlich ein valides JSON-Objekt zurück.

Format:

{
  "id": "doc-001",
  "label": "fraud",
  "fraud_type": "identity_theft",
  "difficulty": "medium",
  "text": "<Echter Name>: ...\\nAgent: ...\\n<Echter Name>: ...",
  "source": "synthetic"
}

WICHTIG:
- Kein Markdown.
- Kein Text außerhalb des JSON.
- Keine zusätzlichen Felder.
- Der JSON-String muss gültig sein."""

class TranscriptSchema(BaseModel):
    id: str
    label: str
    fraud_type: str
    difficulty: str
    text: str
    source: str = "synthetic"

completed_counter = 0

async def test_llm_connection():
    print(f"🔍 Teste Verbindung zum LLM unter {NIM_BASE_URL}...")
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Antworte nur mit 'OK'"}],
            max_tokens=10
        )
        print(f"✅ Verbindung erfolgreich! Antwort: '{response.choices[0].message.content.strip()}'")
    except Exception as e:
        print(f"❌ Verbindungstest fehlgeschlagen: {e}")
        raise SystemExit(1)

async def generate_single_sample(index: int, f_data, f_bench):
    global completed_counter
    doc_id = f"doc-{index:05d}"
    is_fraud = (index % 2 != 0)
    tone = random.choice(TONES)
    length_desc, max_tokens_limit = random.choice(LENGTH_PROMPTS)
    customer_name = random.choice(KUNDEN_NAMEN)
    
    if is_fraud:
        scenario, f_type, diff = random.choice(FRAUD_SCENARIOS)
        user_prompt = (
            f"Erstelle ein Betrugsgespräch (Fraud Call).\n"
            f"Kundenname: {customer_name} (Verwende exakt diesen Namen als Präfix im Dialog statt <Name>).\n"
            f"Szenario: {scenario}\n"
            f"Fraud-Type: {f_type} | Schwierigkeit: {diff}\n"
            f"Stimmung des Kunden: {tone} | Länge: {length_desc}.\n\n"
            f"Generiere das JSON exakt nach Vorgabe."
        )
    else:
        scenario, f_type, diff = random.choice(LEGITIMATE_SCENARIOS)
        user_prompt = (
            f"Erstelle ein legitimes Kundengespräch.\n"
            f"Kundenname: {customer_name} (Verwende exakt diesen Namen als Präfix im Dialog statt <Name>).\n"
            f"Szenario: {scenario}\n"
            f"Fraud-Type: {f_type} | Schwierigkeit: {diff}\n"
            f"Stimmung des Kunden: {tone} | Länge: {length_desc}.\n\n"
            f"Generiere das JSON exakt nach Vorgabe."
        )

    async with semaphore:
        for attempt in range(3):
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
                clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
                clean_text_fixed = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_text)
                
                raw_json = json.loads(clean_text_fixed)

                data = TranscriptSchema(
                    id=doc_id,
                    label=raw_json.get("label", "fraud" if is_fraud else "legitimate"),
                    fraud_type=raw_json.get("fraud_type", f_type),
                    difficulty=raw_json.get("difficulty", diff),
                    text=raw_json.get("text", ""),
                    source="synthetic"
                )

                async with file_lock:
                    f_data.write(data.model_dump_json() + "\n")
                    f_data.flush()
                    
                    f_bench.write(json.dumps({"id": doc_id, "label": data.label}, ensure_ascii=False) + "\n")
                    f_bench.flush()
                    
                    completed_counter += 1
                    if completed_counter % 50 == 0 or completed_counter == NUM_SAMPLES:
                        print(f"⏳ Fortschritt: [{completed_counter}/{NUM_SAMPLES}] ({completed_counter/NUM_SAMPLES*100:.1f}%)")
                return

            except (ValidationError, Exception) as e:
                if attempt == 2:
                    fallback_text = f"{customer_name}: Guten Tag, hier spricht {customer_name}.\nAgent: Guten Tag {customer_name}, wie kann ich Ihnen heute weiterhelfen?"
                    fallback_data = TranscriptSchema(
                        id=doc_id,
                        label="fraud" if is_fraud else "legitimate",
                        fraud_type=f_type if is_fraud else "legitimate",
                        difficulty="medium",
                        text=fallback_text,
                        source="synthetic"
                    )
                    async with file_lock:
                        f_data.write(fallback_data.model_dump_json() + "\n")
                        f_bench.write(json.dumps({"id": doc_id, "label": fallback_data.label}, ensure_ascii=False) + "\n")
                        f_data.flush()
                        f_bench.flush()
                await asyncio.sleep(1 * (attempt + 1))

async def main():
    await test_llm_connection()
    print(f"\n🚀 Starte Generierung von {NUM_SAMPLES} Datensätzen...")
    
    with open(OUTPUT_DATA_FILE, "a", encoding="utf-8") as f_data, \
         open(OUTPUT_BENCHMARK_FILE, "a", encoding="utf-8") as f_bench:
        
        tasks = [generate_single_sample(i, f_data, f_bench) for i in range(1, NUM_SAMPLES + 1)]
        await asyncio.gather(*tasks)

    print(f"\n✅ Fertig!")

if __name__ == "__main__":
    asyncio.run(main())