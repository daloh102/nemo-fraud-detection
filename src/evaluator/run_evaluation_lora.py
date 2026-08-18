import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

VAL_FILE = "/data/nemo-fraud-detection/data/sft/validation_new.jsonl"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "/data/nemo-fraud-detection/data/evaluation/fraud_detection_qlora"

def evaluate_lora_model():
    print("🚀 Lade feingetuntes LoRA-Modell für die Evaluierung...\n")
    
    # Tokenizer und Basismodell laden
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA Adapter laden
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    y_true = []
    y_pred = []
    total = 0
    passed_judge = 0
    
    print(f"📂 Starte Auswertung der Validierungsdaten aus: {VAL_FILE}\n")
    
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            transcript_text = data.get("input", "")
            true_label = data.get("output", "").strip().lower()
            
            if not transcript_text:
                continue
            
            total += 1
            
            # Da das Modell feingetunt ist, reicht ein direkter und sauberer Prompt
            prompt = (
                f"### Instruction:\n"
                f"{transcript_text}\n\n"
                f"### Response:\n"
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.0,
                    do_sample=False
                )
                
            model_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Parsing des Ergebnisses
            if "legitimate" in model_output and "fraud" not in model_output:
                model_answer = "legitimate"
            elif "fraud" in model_output and "legitimate" not in model_output:
                model_answer = "fraud"
            else:
                match = re.search(r'\b(fraud|legitimate)\b', model_output)
                model_answer = match.group(1) if match else "unknown"
            
            y_true.append(true_label)
            y_pred.append(model_answer)
            
            if model_answer == true_label:
                passed_judge += 1

    # Metriken berechnen
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=["fraud", "legitimate"], average="weighted", zero_division=0
    )
    judge_rate = (passed_judge / total) * 100 if total > 0 else 0

    print("\n" + "="*50)
    print("📊 ERGEBNISSE NACH DEM LORA FINE-TUNING:")
    print("="*50)
    print(f"Accuracy: {acc * 100:.2f}% | F1-Score: {f1:.4f} | Match-Rate: {judge_rate:.2f}%")
    print("-" * 50)
    print(classification_report(y_true, y_pred, labels=["fraud", "legitimate"], target_names=["fraud", "legitimate"], zero_division=0))

if __name__ == "__main__":
    evaluate_lora_model()