import os
import json
import random

# Native Importe aus dem NVIDIA NeMo Curator / Data Designer SDK
try:
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.sdg import LLMClient, SyntheticDataGenerator
    NEMO_SDG_AVAILABLE = True
except ImportError:
    NEMO_SDG_AVAILABLE = False
    print("⚠️ NeMo SDG Module nicht gefunden. Nutze Fallback/API-Modus.")

# ==============================================================================
# PFAD-CONFIG MAPPING AUF DEINE ORDNERSTRUKTUR
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_OUT_PATH = os.path.join(BASE_DIR, "data", "raw", "call_transcripts_raw.jsonl")
BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "evaluation", "benchmark_dataset.jsonl")

# Konfiguration: Großes Modell für SDG
SDG_LLM_URL = os.getenv("SDG_LLM_URL", "http://localhost:8000/v1")
SDG_MODEL_NAME = os.getenv("SDG_MODEL_NAME", "meta/llama-3.1-70b-instruct") # An dein großes NIM-Image anpassen!

TOTAL_RECORDS = 10000  # Für erste Testläufe z.B. 100 nutzen, später 10000

NEMO_SYSTEM_PROMPT = (
    "Du bist der NVIDIA NeMo Data Designer für synthetische Dialoggenerierung (SDG).\n"
    "Generiere einen hochrealistischen Telefonmitschnitt (Multi-Turn-Dialog) zwischen einem "
    "Kunden und einer Bank (oder einem Scammer). Der Dialog muss dynamisch sein und zwischen "
    "3 und 10 Turns (Sprecherwechseln) enthalten.\n"
    "Gib AUSSCHLIESSLICH ein valides JSON-Objekt im folgenden Format zurück:\n"
    "{\n  \"transcript\": [\n    {\"speaker\": \"Rolle\", \"text\": \"Inhalt\"}\n  ]\n}"
)

SCENARIOS = {
    "legit": [
        "Kunde möchte den Freistellungsauftrag ändern.",
        "Kreditkarte des Kunden wird an einer Kasse im Ausland abgelehnt.",
        "Kunde fragt nach einer unbekannten Abbuchung auf dem Girokonto.",
        "Kunde möchte eine Vollmacht für ein Familienmitglied einrichten."
    ],
    "fraud": [
        "Scammer gibt sich als IT-Sicherheitsabteilung aus und verlangt SMS-TAN/Kreditkartennummer.",
        "Enkeltrick am Telefon: Betrüger fordert Kaution nach Unfall auf eine ausländische IBAN.",
        "Vishing: Falscher Bankmitarbeiter fordert Verifikation des Online-Bankings."
    ]
}

cities = ["Berlin", "Hamburg", "München", "Köln", "Frankfurt", "Stuttgart"]
pii_samples = [
    "Meine IBAN lautet DE89 3704 0044 0532 0130 00.",
    "Rufen Sie mich unter +49 171 12345678 zurück.",
    "Verwenden Sie die Kreditkarte 4532-1100-8821-9943."
]
garbage_snippets = ["OK", "Ja", "Test 123", "Hallo?", "...", "---"]


def generate_and_contaminate(records_to_generate: int = TOTAL_RECORDS):
    print(f"🚀 Starte NVIDIA NeMo Data Designer Pipeline...")
    print(f"📂 Projekt-Basis: {BASE_DIR}")
    print(f"🤖 Verwende SDG-Modell: {SDG_MODEL_NAME} via {SDG_LLM_URL}")
    print(f"📊 Ziel-Anzahl Datensätze: {records_to_generate}")
    
    os.makedirs(os.path.dirname(RAW_OUT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. SEED PROMPTS & BENCHMARK ERSTELLEN
    # --------------------------------------------------------------------------
    prompt_list = []
    benchmark_dataset = []

    for i in range(records_to_generate):
        call_id = f"CALL_{i:05d}"
        is_fraud = random.random() <= 0.15  # 15% Fraud-Quote
        label = "fraud" if is_fraud else "legit"
        
        base_scenario = random.choice(SCENARIOS[label])
        enriched_scenario = f"{base_scenario} (Ort: {random.choice(cities)}, Betrag: {random.randint(50, 5000)} Euro)."
        
        # Datensatz für das geschützte Evaluation-Benchmark
        benchmark_dataset.append({
            "call_id": call_id,
            "ground_truth": label,
            "scenario": enriched_scenario
        })
        
        # Seed-Dokument für das NeMo DocumentDataset
        prompt_list.append({
            "id": call_id,
            "system_prompt": NEMO_SYSTEM_PROMPT,
            "prompt": f"Generiere einen Dialog zu folgendem Szenario: {enriched_scenario}"
        })

    # --------------------------------------------------------------------------
    # 2. NEMO DATA DESIGNER EXECUTION (GPU LLM INFERENZ)
    # --------------------------------------------------------------------------
    print("⚡ Starte NeMo Data Designer SDG Engine...")
    
    if NEMO_SDG_AVAILABLE:
        client = LLMClient(
            api_url=SDG_LLM_URL,
            model=SDG_MODEL_NAME,  # KORREKTUR: Dynamischer Modellname für das große NIM
            max_tokens=1024,
            temperature=0.7
        )
        generator = SyntheticDataGenerator(client=client)
        
        seed_dataset = DocumentDataset.from_prompt_list(prompt_list)
        generated_dataset = generator.generate(seed_dataset)
        raw_results = generated_dataset.to_dict()
    else:
        print("💡 Fallback/Simulations-Modus für NeMo SDG...")
        raw_results = []
        for p in prompt_list:
            raw_results.append({
                "id": p["id"],
                "response": json.dumps({
                    "transcript": [
                        {"speaker": "Agent", "text": "Guten Tag, Volksbank Kundenservice. Wie kann ich helfen?"},
                        {"speaker": "Customer", "text": f"Hallo, es geht um folgendes Thema: {p['prompt']}"}
                    ]
                })
            })

    # --------------------------------------------------------------------------
    # 3. KONTAMINIERUNG & MÜLL-INJEKTION FÜR DEN CURATOR TEST
    # --------------------------------------------------------------------------
    print("🧬 Injiziere Datenmüll, PII-Daten, Duplikate & Encoding-Fehler...")
    raw_noisy_dataset = []

    for item in raw_results:
        call_id = item.get("id", "CALL_UNKNOWN")
        raw_response = item.get("response", "")
        
        # Parsing des LLM JSON-Outputs
        try:
            parsed = json.loads(raw_response)
            flat_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in parsed["transcript"]])
        except Exception:
            flat_text = f"Agent: Guten Tag.\nAnrufer: Ich habe eine Frage zu meinem Konto."

        dice = random.random()
        
        # Noise-Injektionen
        if dice < 0.08:   # Exaktes Duplikat
            raw_noisy_dataset.append({"call_id": call_id, "text": flat_text})
            raw_noisy_dataset.append({"call_id": f"{call_id}_DUP", "text": flat_text})
            
        elif dice < 0.15: # PII-Daten injizieren
            pii_text = f"{flat_text}\nAnrufer: {random.choice(pii_samples)}"
            raw_noisy_dataset.append({"call_id": call_id, "text": pii_text})
            
        elif dice < 0.20: # Zu kurzer Müll-Text
            raw_noisy_dataset.append({"call_id": call_id, "text": random.choice(garbage_snippets)})
            
        elif dice < 0.23: # Kaputtes UTF-8 Encoding
            broken_text = flat_text.replace("ä", "Ã¤").replace("ö", "Ã¶").replace("ü", "Ã¼")
            raw_noisy_dataset.append({"call_id": call_id, "text": broken_text})
            
        else: # Sauberer Text
            raw_noisy_dataset.append({"call_id": call_id, "text": flat_text})

    # --------------------------------------------------------------------------
    # 4. EXPORT IN DIE ORDNERSTRUKTUR
    # --------------------------------------------------------------------------
    print(f"💾 Schreibe unbereinigte Rohdaten nach: {RAW_OUT_PATH}")
    with open(RAW_OUT_PATH, "w", encoding="utf-8") as f:
        for rec in raw_noisy_dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"💾 Schreibe unberührte Ground-Truth Baseline nach: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        for rec in benchmark_dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ NeMo Data Designer Pipeline erfolgreich ausgeführt! ({len(raw_noisy_dataset)} Einträge erzeugt)")


if __name__ == "__main__":
    # Standardmäßig 50 für den Testlauf, damit es schnell geht. 
    # Später einfach generate_and_contaminate(10000) aufrufen!
    generate_and_contaminate(records_to_generate=50)