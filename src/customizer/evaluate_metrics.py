import json
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/results/fraud_detection_qlora"
TEST_FILE = "/data/nemo-fraud-detection/data/sft/test.jsonl"

def evaluate_detailed_metrics(num_samples=100):
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

    total_samples = min(num_samples, len(test_lines))
    
    fraud_total = 0
    fraud_correct = 0
    legit_total = 0
    legit_correct = 0

    print(f"\n🚀 Analysiere {total_samples} Datensätze...\n")

    for i, example in enumerate(test_lines[:total_samples]):
        user_input = example.get("input", "")
        expected_output = example.get("output", "").strip().lower()

        gt = "fraud" if "fraud" in expected_output else "legitimate"

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
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = decoded_output.split("### Response:")[-1].strip().lower()

        if "fraud" in response_only and "legitimate" not in response_only:
            pred = "fraud"
        elif "legitimate" in response_only or "legitim" in response_only:
            pred = "legitimate"
        else:
            pred = "other"

        if gt == "fraud":
            fraud_total += 1
            if pred == "fraud":
                fraud_correct += 1
        else:
            legit_total += 1
            if pred == "legitimate":
                legit_correct += 1

    print("="*50)
    print("📊 DETAIL-AUSWERTUNG DER KLASSIFIKATION")
    print("="*50)
    if fraud_total > 0:
        print(f"🚨 Fraud-Fälle erkannt: {fraud_correct} von {fraud_total} ({100 * fraud_correct / fraud_total:.1f}%)")
    else:
        print("🚨 Keine Fraud-Fälle in den Testdaten gefunden.")
        
    if legit_total > 0:
        print(f"✅ Legitimate-Fälle erkannt: {legit_correct} von {legit_total} ({100 * legit_correct / legit_total:.1f}%)")
    else:
        print("✅ Keine Legitimate-Fälle in den Testdaten gefunden.")
        
    total_correct = fraud_correct + legit_correct
    print(f"📈 Gesamte Genauigkeit: {100 * total_correct / total_samples:.1f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_detailed_metrics(100)