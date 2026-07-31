import json
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Logging reduzieren, damit die Tabelle übersichtlich bleibt
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/results/fraud_detection_qlora"
TEST_FILE = "/data/nemo-fraud-detection/data/sft/test.jsonl"

def evaluate_table(num_samples=100):
    print("📥 Lade Basismodell und Adapter...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print(f"📂 Lade Testdaten aus: {TEST_FILE}")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_lines = [json.loads(line) for line in f]

    # Auf max. verfügbare oder 100 limitieren
    total_samples = min(num_samples, len(test_lines))
    
    results = []

    print(f"\n🚀 Starte Auswertung von {total_samples} Datensätzen...\n")

    for i, example in enumerate(test_lines[:total_samples]):
        user_input = example.get("input", "")
        expected_output = example.get("output", "").strip().lower()

        # Schärferer Prompt, um eine klare Klassifikation zu erzwingen
        prompt = (
            f"### Instruction:\n"
            f"Klassifiziere den folgenden Dialog ausschließlich als 'fraud' oder 'legitimate'. "
            f"Antworte nur mit einem dieser beiden Wörter.\n\n"
            f"Dialog:\n{user_input}\n\n"
            f"### Response:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Sehr kurz halten, nur für das Label
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = decoded_output.split("### Response:")[-1].strip().lower()

        # Auf die Kernwörter filtern
        if "fraud" in response_only and "legitimate" not in response_only:
            pred = "fraud"
        elif "legitimate" in response_only or "legitim" in response_only:
            pred = "legitimate"
        else:
            pred = response_only[:20]  # Fallfalls es abweicht

        # Ground Truth bereinigen (falls dort ganze Sätze stehen)
        gt = "fraud" if "fraud" in expected_output else ("legitimate" if "legitim" in expected_output else expected_output)

        match_status = "✅ KORREKT" if gt == pred else "❌ FALSCH"
        results.append((i + 1, gt, pred, match_status))

    # Tabellen-Ausgabe
    print("\n" + "="*65)
    print(f"{'Nr.':<5} | {'Ground Truth':<15} | {'Modell-Vorhersage':<18} | {'Status':<10}")
    print("="*65)
    
    correct_count = 0
    for r in results:
        print(f"{r[0]:<5} | {r[1]:<15} | {r[2]:<18} | {r[3]:<10}")
        if "KORREKT" in r[3]:
            correct_count += 1

    print("="*65)
    accuracy = (correct_count / total_samples) * 100
    print(f"📊 Gesamtergebnis: {correct_count} von {total_samples} korrekt ({accuracy:.1f}% Genauigkeit)")
    print("="*65 + "\n")

if __name__ == "__main__":
    evaluate_table(100)