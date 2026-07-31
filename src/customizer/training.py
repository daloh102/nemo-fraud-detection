import os
import sys
import torch
import logging
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import get_peft_model, LoraConfig
from huggingface_hub import snapshot_download
from datasets import load_dataset
from trl import SFTTrainer

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = "/data/nemo-fraud-detection"

def validate_prerequisites(cfg: DictConfig) -> None:
    logging.info("🔍 Überprüfe Voraussetzungen für das Training...")

    if not torch.cuda.is_available():
        raise RuntimeError("❌ Keine CUDA-fähige GPU gefunden!")
    
    vram_free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    logging.info(f"✅ GPU erkannt: {torch.cuda.get_device_name(0)} (Freier VRAM: {vram_free_gb:.2f} GB)")

    model_path = cfg.model.get("restore_from_path", None)
    if not model_path:
        raise ValueError("❌ In der Konfiguration fehlt 'cfg.model.restore_from_path'!")

    logging.info("✅ Voraussetzungen erfolgreich geprüft.")


def run_training(cfg: DictConfig) -> bool:
    try:
        validate_prerequisites(cfg)

        model_path = cfg.model.restore_from_path
        if model_path.startswith("hf://"):
            model_path = model_path.replace("hf://", "")

        if not os.path.exists(model_path) and not "/" in model_path:
            logging.info(f"📥 Lade Hugging Face Modell '{model_path}' herunter...")
            model_path = snapshot_download(repo_id=model_path)

        logging.info(f"📥 Initialisiere QLoRA Modell von: {model_path}...")

        # 1. QLoRA Konfiguration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # 2. LoRA Adapter Konfiguration
        peft_config = LoraConfig(
            r=cfg.model.get("peft", {}).get("r", 16),
            lora_alpha=cfg.model.get("peft", {}).get("alpha", 32),
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "w1", "w2"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 3. Datensätze laden (aus YAML)
        # 3. Datensätze laden (aus YAML - robust für Listen und Strings)
        train_file = OmegaConf.to_container(cfg.model.data.train_ds.file_names, resolve=True)
        if isinstance(train_file, list):
            train_file = train_file[0]
            
        if not os.path.isabs(train_file):
            train_file = os.path.join(PROJECT_ROOT, train_file)

        logging.info(f"📂 Lade Trainingsdaten von: {train_file}")
        dataset = load_dataset("json", data_files={"train": train_file})

        save_path = cfg.model.get("save_to", "results/fraud_detection_qlora")
        abs_save_path = save_path if os.path.isabs(save_path) else os.path.join(PROJECT_ROOT, save_path)

        training_args = TrainingArguments(
            output_dir=abs_save_path,
            per_device_train_batch_size=cfg.model.get("micro_batch_size", 2),
            gradient_accumulation_steps=cfg.model.get("gradient_accumulation_steps", 4),
            learning_rate=cfg.model.get("optim", {}).get("lr", 2e-4),
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            optim="paged_adamw_8bit",
            max_steps=cfg.model.get("max_steps", 100),
            report_to="none"
        )

        # 4. SFTTrainer starten
        def formatting_func(example):
            if "input" in example and "output" in example:
                return [f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"]
            elif "text" in example:
                return [example["text"]]
            else:
                return [str(list(example.values())[0])]

        # SFTTrainer mit Formatierungsfunktion starten
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            formatting_func=formatting_func,
            max_seq_length=2048,
            tokenizer=tokenizer,
            args=training_args,
        )

        logging.info("🚀 Starte Fine-Tuning (Training)...")
        trainer.train()
        # 5. Speichern des finalen Adapters
        trainer.model.save_pretrained(abs_save_path)
        tokenizer.save_pretrained(abs_save_path)
        logging.info(f"✅ Training abgeschlossen! Modell gespeichert unter: {abs_save_path}")
        return True

    except Exception as e:
        logging.error(f"\n💥 Fehler während des Trainings: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.isabs(config_file):
            config_file = os.path.join(PROJECT_ROOT, config_file)
            
        logging.info(f"📖 Lade Konfiguration von: {config_file}")
        cfg = OmegaConf.load(config_file)
        
        success = run_training(cfg)
        sys.exit(0 if success else 1)
    else:
        print("❌ Bitte Konfigurationspfad angeben: python3 training.py configs/customizer/qlora.yaml")
        sys.exit(1)