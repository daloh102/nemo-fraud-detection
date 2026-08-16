import json
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/results/fraud_detection_qlora"
TEST_FILE = "/data/nemo-fraud-detection/data/sft/test.jsonl"

def evaluate_raw(num_samples=5):
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

    print(f"\n🔍 Zeige rohe Modell-Antworten für die ersten {total_samples} Testbeispiele:\n" + "="*60)

    for i, example in enumerate(test_lines[:total_samples]):
        user_input = example.get("input", "")
        expected_output = example.get("output", "").strip().lower()
        gt = "fraud" if "fraud" in expected_output else "legitimate"

        prompt = f"### Instruction:\nAnalysiere den folgenden Dialog und entscheide, ob es sich um Betrug oder einen legitimen Vorgang handelt:\n{user_input}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = decoded_output.split("### Response:")[-1].strip()

        print(f"\n[Beispiel {i+1}]")
        print(f"🎯 Erwartet (Ground Truth): {gt.upper()}")
        print(f"🤖 Rohe Modell-Antwort:\n{response_only}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate_raw(5)