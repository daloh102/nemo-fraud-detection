import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/results/fraud_detection_qlora"
TEST_FILE = "/data/nemo-fraud-detection/data/sft/test.jsonl"
OUTPUT_FILE = "/data/nemo-fraud-detection/results/evaluation_english_prompt_10.jsonl"

PROMPT_TEMPLATE = (
    "Phone call transcript: {text}\n\n"
    "Question: Based on the phone call transcript above, is it a fraudulent call or a legitimate call?\n\n"
    "Answer: "
)

def evaluate_with_english_prompt(num_samples=10):
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
    
    results = []
    file_records = []
    fraud_total = 0
    fraud_correct = 0
    legit_total = 0
    legit_correct = 0

    print(f"\n🚀 Starte Auswertung mit englischem Prompt für {total_samples} Datensätze...\n")

    for i, example in enumerate(test_lines[:total_samples]):
        user_input = example.get("input", "")
        expected_output = example.get("output", "").strip().lower()
        gt = "fraud" if "fraud" in expected_output else "legitimate"

        prompt = PROMPT_TEMPLATE.format(text=user_input)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,  
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = decoded_output.split("Answer:")[-1].strip().lower()

        if "fraud" in response_only and "legitimate" not in response_only:
            pred = "fraud"
        elif "legitimate" in response_only or "legitim" in response_only:
            pred = "legitimate"
        else:
            pred = "fraud" if "fraud" in response_only else "legitimate"

        match = (gt == pred)
        status_str = "✅ KORREKT" if match else "❌ FALSCH"
        
        results.append((i + 1, gt, pred, status_str))

        file_records.append({
            "id": i + 1,
            "input": user_input,
            "ground_truth": gt,
            "prediction": pred,
            "status": "CORRECT" if match else "FALSE",
            "model_response": response_only
        })

        if gt == "fraud":
            fraud_total += 1
            if match: fraud_correct += 1
        else:
            legit_total += 1
            if match: legit_correct += 1

    print(f"💾 Speichere Ergebnisse in: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for record in file_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "="*65)
    print(f"{'Nr.':<5} | {'Ground Truth':<15} | {'Modell-Vorhersage':<18} | {'Status':<10}")
    print("="*65)
    
    for r in results:
        print(f"{r[0]:<5} | {r[1]:<15} | {r[2]:<18} | {r[3]:<10}")

    total_correct = fraud_correct + legit_correct
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

    print("="*65)
    print(f"📊 GESAMT-AUSWERTUNG ({total_samples} Datensätze):")
    print(f"🚨 Fraud-Erkennung:     {fraud_correct} von {fraud_total} ({ (fraud_correct/fraud_total)*100 if fraud_total>0 else 0:.1f}%)")
    print(f"✅ Legitimate-Erkennung: {legit_correct} von {legit_total} ({ (legit_correct/legit_total)*100 if legit_total>0 else 0:.1f}%)")
    print(f"📈 Gesamtergebnis:       {total_correct} von {total_samples} korrekt ({accuracy:.1f}% Genauigkeit)")
    print("="*65 + "\n")

if __name__ == "__main__":
    evaluate_with_english_prompt(10)