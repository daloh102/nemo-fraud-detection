import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/results/fraud_detection_qlora"

def run_inference():
    logging.info("📥 Lade Basismodell und Tokenizer...")
    
    # 4-Bit Konfiguration analog zum Training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    logging.info(f"🔗 Lade trainierten QLoRA-Adapter von: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Test-Prompt (Passe diesen Text an einen typischen Betrugsfall aus deinen Daten an)
    test_instruction = "Prüfe die folgende Transaktion auf Betrugsmerkmale: Transaktionsbetrag 9999.00 EUR, Land: Auslandsüberweisung High-Risk."
    
    prompt = f"### Instruction:\n{test_instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    logging.info("🚀 Generiere Vorhersage...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print(decoded_output)
    print("="*50 + "\n")

if __name__ == "__main__":
    run_inference()