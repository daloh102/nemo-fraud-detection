import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/results/fraud_detection_qlora"
TEST_FILE = "/data/nemo-fraud-detection/data/sft/test.jsonl"

def evaluate_with_reasoning(num_samples=20):
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
    
    correct_count = 0

    print(f"\n🚀 Starte qualitative Auswertung von {total_samples} Beispielen (mit Antwort-Text)...\n" + "="*70)

    for i, example in enumerate(test_lines[:total_samples]):
        user_input = example.get("input", "")
        expected_output = example.get("output", "").strip().lower()
        gt = "fraud" if "fraud" in expected_output else "legitimate"

        # Natürlicher Prompt im gelernten SFT-Format
        prompt = f"### Instruction:\nPrüfe die folgende Transaktion bzw. das Gespräch auf Betrugsmerkmale:\n{user_input}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Genug Platz für eine kurze Erklärung
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = decoded_output.split("### Response:")[-1].strip()

        # Automatische Erkennung, ob das Modell im Text zu dem Schluss kommt, dass es Betrug oder legal ist
        response_lower = response_only.lower()
        
        # Einfache Heuristik zur Klassifikation des Textes
        if any(w in response_lower for w in ["betrug", "fraud", "vorsichtig", "gefährlich", "warnung", "kriminell"]):
            pred = "fraud"
        else:
            pred = "legitimate"

        match = gt == pred
        if match:
            correct_count += 1

        status_icon = "✅ KORREKT" if match else "❌ FALSCH"

        print(f"\n[Beispiel {i+1}] - Status: {status_icon}")
        print(f"🎯 Ground Truth: {gt.upper()} | 🤖 Modell-Tendenz erkannt als: {pred.upper()}")
        print(f"💬 Modell-Antwort:\n{response_only}")
        print("-" * 70)

    print(f"\n📊 Zwischenergebnis (erste {total_samples} Beispiele): {correct_count} von {total_samples} korrekt ({(correct_count/total_samples)*100:.1f}%)")
    print("="*70 + "\n")

if __name__ == "__main__":
    evaluate_with_reasoning(20)