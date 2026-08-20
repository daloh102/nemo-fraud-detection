import json
import shutil
import subprocess
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. ENVIRONMENT & GPU CHECK
# ==========================================
def check_environment():
    print("=== Überprüfe Umgebung und GPU ===")
    
    # nvidia-smi Check
    try:
        smi_output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True, text=True, check=True
        ).stdout
        print(smi_output)
    except Exception as e:
        print(f"Warnung bei nvidia-smi: {e}")

    # PyTorch CUDA Check
    print("CUDA verfügbar:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
        print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    assert torch.cuda.is_available(), "Keine GPU sichtbar — stelle sicher, dass der Container mit --gpus all gestartet wurde."

    # automodel CLI Check
    automodel_path = shutil.which("automodel")
    assert automodel_path, "automodel CLI nicht gefunden — prüfe, ob nemo_automodel korrekt installiert ist."
    print("automodel CLI gefunden unter:", automodel_path)


# ==========================================
# 2. DATA PREPARATION
# ==========================================
PROMPT_TEMPLATE = (
    "Phone call transcript: {text}\n\n"
    "Question: Based on the phone call transcript above, is it a fraudulent call or a legitimate call?\n\n"
    "Answer: "
)

def prepare_data():
    print("\n=== Bereite Daten vor ===")
    def to_record(entry):
        return {
            "input": PROMPT_TEMPLATE.format(text=entry["input"]),
            "output": entry["output"]
        }

    data_files = ["data/raw/train.jsonl", "data/raw/test.jsonl", "data/raw/validation.jsonl"]
    for data_path in data_files:
        path_obj = Path(data_path)
        if not path_obj.exists():
            print(f"Überspringe {data_path}, da nicht vorhanden.")
            continue
            
        with open(path_obj) as raw_data_file:
            records = [to_record(json.loads(line)) for line in raw_data_file]
            processed_data_path = Path("data") / path_obj.name
            processed_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(processed_data_path, "w") as processed_data_file:
                for record in records:
                    processed_data_file.write(json.dumps(record) + "\n")
        print(oges := f"Verarbeitet und gespeichert: {processed_data_path}")


# ==========================================
# 3. TRAINING
# ==========================================
def run_training(smoke_test=False):
    print("\n=== Starte Training ===")
    config_path = "configs/qwen2_1p5b_sft.yaml"
    
    if smoke_test:
        print("Führe Smoke-Test aus (50 Schritte)...")
        subprocess.run([
            "automodel", config_path,
            "--step_scheduler.max_steps", "50",
            "--checkpoint.checkpoint_dir", "results/qwen2_1p5b_sft_smoke_test/checkpoints"
        ], check=True)
    else:
        print("Führe vollständiges Training aus...")
        subprocess.run(["automodel", config_path], check=True)


# ==========================================
# 4. EVALUATION HELPER FUNCTIONS
# ==========================================
def load_model(checkpoint: str):
    is_local_peft_adapter = Path(checkpoint).is_dir() and (Path(checkpoint) / "adapter_config.json").exists()
    if is_local_peft_adapter:
        from peft import AutoPeftModelForCausalLM
        print(f"Lade PEFT/LoRA Adapter von {checkpoint}")
        model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda")
        tokenizer_source = checkpoint
    else:
        print(f"Lade Basismodell {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda")
        tokenizer_source = checkpoint

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer

def generate_batch(model, tokenizer, prompts, max_new_tokens=16):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completions = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def generate_predictions(checkpoint: str, input_jsonl: str, output_jsonl: str, limit: int = None, max_batch_size: int = 8):
    with open(input_jsonl) as f:
        records = [json.loads(line) for line in f]
    if limit:
        records = records[:limit]

    model, tokenizer = load_model(checkpoint)
    predictions = []
    batch_timings = []
    
    for i in range(0, len(records), max_batch_size):
        batch = records[i : i + max_batch_size]
        prompts = [r["input"] for r in batch]
        start = time.perf_counter()
        predictions.extend(generate_batch(model, tokenizer, prompts))
        batch_timings.append({"n": len(batch), "seconds": time.perf_counter() - start})

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record, prediction in zip(records, predictions):
            out = dict(record)
            out["prediction"] = prediction.strip()
            f.write(json.dumps(out) + "\n")

    print(f"Vorhersagen geschrieben nach {output_jsonl}")


# ==========================================
# 5. EVALUATION RUNNER
# ==========================================
def evaluate_models():
    print("\n=== Starte Evaluierung ===")
    BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
    RESULTS_DIR = Path("results")
    RUN_NAME = "qwen2-1p5b-sft"

    checkpoint_path = RESULTS_DIR / RUN_NAME / "checkpoints" / "LATEST" / "model"
    if (checkpoint_path / "consolidated").exists():
        checkpoint_path = checkpoint_path / "consolidated"
        
    assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} nicht gefunden — bitte erst das Training ausführen."

    EVAL_N = 200
    base_output = str(RESULTS_DIR / "eval_base_predictions.jsonl")
    finetuned_output = str(RESULTS_DIR / "eval_finetuned_predictions.jsonl")
    test_jsonl = str(Path("data") / "test.jsonl")

    # Generieren
    generate_predictions(BASE_MODEL, test_jsonl, base_output, EVAL_N)
    generate_predictions(str(checkpoint_path), test_jsonl, finetuned_output, EVAL_N)

    def normalize(text):
        text = str(text).strip().lower()
        if "fraud" in text:
            return "fraud"
        if "legitimate" in text:
            return "legitimate"
        return "unparsed"

    def load_preds(output_path):
        with open(output_path) as f:
            rows = [json.loads(line) for line in f]
        return [normalize(r["output"]) for r in rows], [normalize(r["prediction"]) for r in rows]

    true_labels, base_preds = load_preds(base_output)
    _, finetuned_preds = load_preds(finetuned_output)

    results = [("Base model (zero-shot)", base_preds), ("Fine-tuned", finetuned_preds)]

    for label, preds in results:
        print(f"\n=== {label} ===")
        print(classification_report(true_labels, preds, labels=["fraud", "legitimate"], zero_division=0))
        print("Confusion matrix (rows=true, cols=predicted, order=[fraud, legitimate]):")
        print(confusion_matrix(true_labels, preds, labels=["fraud", "legitimate"]))
        unparsed = sum(1 for p in preds if p == "unparsed")
        if unparsed:
            print(f"({unparsed} Vorhersagen konnten nicht als fraud/legitimate geparst werden)")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    check_environment()
    prepare_data()
    
    # Wähle hier aus, ob du einen Smoke-Test (True) oder Voll-Training (False) machen willst:
    run_training(smoke_test=True) 
    
    evaluate_models()