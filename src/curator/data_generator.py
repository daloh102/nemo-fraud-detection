import json
import random
import re
from openai import OpenAI

# 1. Verbindung zum lokalen NIM
client = OpenAI(
    base_url="http://172.31.18.77:8000/v1",
    api_key="not-needed"
)

MODEL_NAME = "meta/llama-3.1-8b-instruct"
OUTPUT_DATA_FILE = "fraud_call_transcripts.jsonl"
OUTPUT_BENCHMARK_FILE = "fraud_call_benchmark.jsonl"
NUM_SAMPLES = 5000  # Anzahl der zu generierenden Datensätze

# 2. Erweiterter Szenarien-Katalog
FRAUD_SCENARIOS = [
    "Falscher Bankmitarbeiter: TAN / 2FA-Freigabe für angebliche Test-Überweisung erschleichen",
    "SMS / Push-TAN Phishing: Kundendienst behauptet, Push-App müsse neu aktiviert werden",
    "Karten-Sperrung: Anrufer fordert Kreditkartennummer & CVC wegen angeblichem Missbrauch",
    "Enkeltrick / Schockanruf: Kaution für angeblichen Verkehrsunfall des Kindes",
    "Fake-Polizist: Vorwand von Einbrüchen in der Nachbarschaft; Geld soll sichergestellt werden",
    "Fake-Notar / Behörde: Angebliches Erbe verlangt sofortige Bearbeitungsgebühr per Überweisung",
    "Tech-Support (Microsoft / Apple): Fernzugriffs-Software (AnyDesk / TeamViewer) installieren lassen",
    "Crypto-Investment-Betrug: Versprechen von extrem hohen Renditen bei Ersteinzahlung",
    "Phantom-Gewinnspiel: Gewinnfreischaltung erst nach Kauf von Gutscheinkarten (GPay/Apple)",
    "Fake-Jobangebot: Anrufer verlangt Eröffnung eines Kontos zur 'Identitätsprüfung'"
]

LEGITIMATE_SCENARIOS = [
    "Kontostand & letzte Buchungen abfragen",
    "Dauerauftrag einrichten, ändern oder löschen",
    "Limits für Online-Überweisungen oder EC-Karte vorübergehend erhöhen",
    "Adresse, Telefonnummer oder E-Mail-Adresse nach Umzug ändern",
    "Kreditkarte nach Verlust im Ausland sofort sperren lassen",
    "Neubestellung einer beschädigten Girocard",
    "Nachfrage zu Konditionen von Festgeld / Tagesgeldkonto",
    "Erstanmeldung & Registrierung für die Mobile-Banking-App",
    "Rückfrage zu einer unbekannten Abbuchung (stellt sich als legitim heraus)",
    "Nachfragen zur Zusendung des Jahressteuerbescheids"
]

TONES = [
    "sehr drängend und autoritär", 
    "verwirrt und unsicher", 
    "extrem freundlich und professionell", 
    "panisch und aufgeregt", 
    "sachlich und neutral"
]

system_prompt = (
    "Du bist ein spezialisierter Data-Generator für Security- & Fraud-Detection-Modelle.\n"
    "Deine Aufgabe ist es, realistische Dialog-Transkripte von Telefonaten zwischen einem Kunden und einem Agenten (bzw. Anrufer) zu erzeugen.\n\n"
    "WICHTIG:\n"
    "1. Der Dialog MUSS immer klar erkennen lassen, wer spricht. Verwende im Text strikt die Präfixe 'Kunde:' und 'Agent:' (oder 'Anrufer:').\n"
    "2. Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt ohne Markdown-Formatierung:\n"
    "{\n"
    '  "id": "doc-XXX",\n'
    '  "text": "Kunde: ... \\nAgent: ... \\nKunde: ...",\n'
    '  "source": "custom"\n'
    "}"
)

print(f"🚀 Starte Generierung von {NUM_SAMPLES} Datensätzen...")

# 3. Haupt-Schleife zur Erzeugung der Datensätze
with open(OUTPUT_DATA_FILE, "w", encoding="utf-8") as f_data, \
     open(OUTPUT_BENCHMARK_FILE, "w", encoding="utf-8") as f_bench:

    for i in range(1, NUM_SAMPLES + 1):
        doc_id = f"doc-{i:05d}"  # Erzeugt doc-00001 bis doc-05000
        
        is_fraud = (i % 2 != 0)  # Abwechselnd Fraud und Legitimate
        label = "fraud" if is_fraud else "legitimate"
        tone = random.choice(TONES)
        
        if is_fraud:
            scenario = random.choice(FRAUD_SCENARIOS)
            user_prompt = (
                f"Generiere ein BETRUGSGESPRAECH (Fraud Call).\n"
                f"Szenario: {scenario}.\n"
                f"Gesprächsatmosphäre/Stimmung: {tone}.\n"
                f"Kennzeichne die Sprecher eindeutig mit 'Kunde:' und 'Agent:' (oder 'Anrufer:')."
            )
        else:
            scenario = random.choice(LEGITIMATE_SCENARIOS)
            user_prompt = (
                f"Generiere ein LEGITIMES, normales Bankgespräch (Legitimate Call).\n"
                f"Szenario: {scenario}.\n"
                f"Gesprächsatmosphäre/Stimmung: {tone}.\n"
                f"Kennzeichne die Sprecher eindeutig mit 'Kunde:' und 'Agent:'."
            )

        # Fortschritts-Anzeige alle 50 Datensätze
        if i % 50 == 0 or i == 1:
            print(f"⏳ Fortschritt: [{i}/{NUM_SAMPLES}] ({i/NUM_SAMPLES*100:.1f}%)")

        try:
            # Temperatur leicht variieren (0.7 bis 0.95) für natürliche Vielfalt
            temp = round(random.uniform(0.7, 0.95), 2)

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temp,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )

            raw_text = response.choices[0].message.content.strip()
            
            # Schritt A: Markdown-Codeblocks entfernen
            clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()

            # Schritt B: Fehlerhafte Backslashes reparieren (\uXXXX Fehler abfangen)
            clean_text_fixed = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', clean_text)

            # Schritt C: Robustes JSON-Parsing mit mehrstufigem Fallback
            try:
                json_data = json.loads(clean_text_fixed)
                transcript_text = json_data.get("text", clean_text_fixed)
            except json.JSONDecodeError:
                # Falls JSON wegen tief sitzender Formatierungsfehler schlägt:
                # Wir entfernen problematische Backslashes direkt aus dem Text
                transcript_text = clean_text.replace('\\', '/')

            # 4. Transkript schreiben & sofort spülen (flush)
            formatted_data = {
                "id": doc_id,
                "text": transcript_text,
                "source": "custom"
            }
            f_data.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
            f_data.flush()

            # 5. Benchmark schreiben & sofort spülen
            benchmark_entry = {
                "id": doc_id,
                "label": label
            }
            f_bench.write(json.dumps(benchmark_entry, ensure_ascii=False) + "\n")
            f_bench.flush()

        except Exception as e:
            # Netzwerkfehler oder Unerwartetes abfangen
            print(f"⚠️ Unerwarteter Systemfehler bei {doc_id}: {e}")
            fallback_text = "Kunde: Guten Tag.\nAgent: Guten Tag. (Fehler bei Modellgenerierung)"
            f_data.write(json.dumps({"id": doc_id, "text": fallback_text, "source": "custom"}, ensure_ascii=False) + "\n")
            f_bench.write(json.dumps({"id": doc_id, "label": label}, ensure_ascii=False) + "\n")
            f_data.flush()
            f_bench.flush()

print(f"\n✅ Fertig! {NUM_SAMPLES} Datensätze wurden erfolgreich erzeugt.")