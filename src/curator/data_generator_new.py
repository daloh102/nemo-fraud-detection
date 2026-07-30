import asyncio
import os
import json
import random
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# 1. Konfiguration über Environment Variables (Fallback auf Default)
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "http://172.31.18.77:8000/v1")
MODEL_NAME = "meta/llama-3.1-8b-instruct"
OUTPUT_DATA_FILE = "fraud_call_transcripts.jsonl"
OUTPUT_BENCHMARK_FILE = "fraud_call_benchmark.jsonl"
NUM_SAMPLES = 5000
CONCURRENCY_LIMIT = 15  # Max 15 parallele Anfragen an das NIM

client = AsyncOpenAI(base_url=NIM_BASE_URL, api_key="not-needed")
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

FRAUD_SCENARIOS = [...]
LEGITIMATE_SCENARIOS = [...]
TONES = [...]

# Structured Output Schema definieren
class TranscriptSchema(BaseModel):
    id: str
    text: str = Field(description="Der komplette Gesprächsverlauf zwischen Kunde und Agent")
    source: str = "custom"

async def generate_single_sample(index: int, f_data, f_bench):
    doc_id = f"doc-{index:05d}"
    is_fraud = (index % 2 != 0)
    label = "fraud" if is_fraud else "legitimate"
    tone = random.choice(TONES)
    scenario = random.choice(FRAUD_SCENARIOS if is_fraud else LEGITIMATE_SCENARIOS)

    prompt = (
        f"Generiere ein {'BETRUGSGESPRAECH' if is_fraud else 'LEGITIMES Bankgespräch'}.\n"
        f"Szenario: {scenario}\nStimmung: {tone}\n"
        f"Nutze strikt die Präfixe 'Kunde:' und 'Agent:'."
    )

    # Begrenzung der parallelen Anfragen (schont die GPU/NIM)
    async with semaphore:
        for attempt in range(3):  # Bis zu 3 Retries
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Du erzeugst Dialog-Transkripte im JSON-Format mit den Feldern 'id', 'text', 'source'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=round(random.uniform(0.7, 0.95), 2),
                    response_format={"type": "json_object"}
                )
                
                raw = json.loads(response.choices[0].message.content)
                # Validierung über Pydantic
                data = TranscriptSchema(id=doc_id, text=raw.get("text", ""))

                # Schreiben in Dateien
                f_data.write(data.model_dump_json() + "\n")
                f_bench.write(json.dumps({"id": doc_id, "label": label}) + "\n")
                return

            except Exception as e:
                if attempt == 2:
                    print(f"❌ Fehler bei {doc_id} nach 3 Versuchen: {e}")
                await asyncio.sleep(1)

async def main():
    print(f"🚀 Starte asynchrone Generierung von {NUM_SAMPLES} Datensätzen...")
    with open(OUTPUT_DATA_FILE, "a", encoding="utf-8") as f_data, \
         open(OUTPUT_BENCHMARK_FILE, "a", encoding="utf-8") as f_bench:
        
        tasks = [generate_single_sample(i, f_data, f_bench) for i in range(1, NUM_SAMPLES + 1)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())