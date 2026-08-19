"""
================================================================================
Projekt:        NeMo Fraud Detection
Skript-Name:    evaluate_lora.py
Beschreibung:   Lokale Evaluierung des feinabgestimmten LoRA-Adapters 
                (basierend auf Llama-3.1-8B-Instruct) für die Betrugserkennung.

Funktionsumfang:
    1. Modell-Initialisierung: Lädt das Basismodell und verknüpft den trainierten LoRA-Adapter.
    2. Inferenz: Generiert Antworten lokal via PyTorch/Transformers auf Basis der Validierungstranskripte.
    3. Parsing & Validierung: Mappt Ground-Truth-Labels und vergleicht sie robust mit der Modellantwort.
    4. Metriken & Tracking: Berechnet die Genauigkeit (Accuracy) und loggt 
       die Ergebnisse an Weights & Biases (wandb).

Eingabedateien:
    - Validierungsdaten: /data/nemo-fraud-detection/data/sft/validation.jsonl
    - LoRA-Adapter:      /data/nemo-fraud-detection/data/evaluation/fraud_detection_qlora

Autor:          Daniel Lohmann
Datum:          2026
Erfolgreich getestet am: 19.08.2026
================================================================================
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import wandb

# Pfade
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/data/evaluation/fraud_detection_qlora"
VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation.jsonl"

def run_lora_evaluation():
    print("🚀 Lade Basis-Modell und LoRA-Adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    wandb.init(project="fraud-detection", name="lora-evaluation")
    correct, total = 0, 0

    with open(VAL_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            input_text = f"### Instruction:\n{data['input']}\n\n### Response:\n"
            
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1].strip().lower()
            
            true_label = "fraud" if "betrug" in data['output'].lower() else "legitimate"
            model_answer = "fraud" if "fraud" in response else "legitimate"
            
            is_correct = (true_label == model_answer)
            if is_correct: correct += 1
            total += 1
            print(f"Erwartet: {true_label} | Erkannt: {model_answer} | Korrekt: {is_correct}")

    accuracy = (correct / total) * 100
    print(f"\n🎯 Lora-Evaluierungs-Genauigkeit: {accuracy:.2f}%")
    wandb.log({"eval/accuracy": accuracy})
    wandb.finish()

if __name__ == "__main__":
    run_lora_evaluation()