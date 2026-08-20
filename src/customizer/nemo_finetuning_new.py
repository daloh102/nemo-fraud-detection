"""
Funktionsbeschreibung der Megatron-GPT Fine-Tuning-Pipeline:

Dieser Quellcode implementiert eine automatisierte und robuste Trainings-Pipeline zur Durchführung 
eines Supervised Fine-Tunings (SFT) für große Sprachmodelle (speziell Llama-3.2-3B) unter Verwendung 
des NVIDIA NeMo-Frameworks und Megatron-LM. Der Prozess gliedert sich in folgende logische Hauptschritte:

*   Infrastruktur & Vorbereitung: Zunächst werden essenzielle Dateipfade (Basismodell und Datensätze im JSONL-Format) 
    definiert. Um Konflikte mit alten Checkpoints und inkonsistenten Zuständen zu vermeiden, wird der 
    bestehende Experiment-Ordner vor jedem Start automatisch bereinigt.

*   Modellkonfiguration & Hardware-Setup: 
    *   Modell-Restaurierung: Das vortrainierte Basismodell wird über den angegebenen Pfad (`.nemo`) geladen.
    *   Hardware-Allokation: Das Training wird für den Betrieb auf einer einzelnen GPU (`devices=1`, `num_nodes=1`) 
        konfiguriert, wobei Tensor- und Pipeline-Parallelität explizit auf Single-GPU-Betrieb eingestellt sind.

*   Logging & Experiment-Management:
    *   Weights & Biases (W&B): Der standardmäßige TensorBoard-Logger wird deaktiviert, um Konflikte zu vermeiden, 
        während der W&B-Logger für ein zentrales Tracking von Metriken und Loss-Verläufen aktiv eingebunden wird.
    *   Stabilitäts-Fix: Zur Vermeidung von internen Typ- und Protobuf-Fehlern in PyTorch Lightning wird der 
        spezifische TFLOPs-Berechnungs-Callback (`log_tflops_per_sec_per_gpu=False`) gezielt unterdrückt.

*   Dataset-Schema & Prompt-Templating: Unter Verwendung des Megatron-Datenformats werden die Trainings- 
    und Validierungsdaten eingebunden. Über ein einheitliches Prompt-Template (`{input}{output}`) wird die 
    strukturierte Einspeisung der SFT-Daten für das Sprachmodell sichergestellt.

Zusammenfassend dient das Skript dazu, ein stabiles, fehlerfreies und vollständig überwachtes Fine-Tuning 
von LLMs für domänenspezifische Aufgaben (wie Fraud Detection) zu gewährleisten und die Ergebnisse 
zuverlässig zu protokollieren.

Autor:         Daniel Lohmann
Datum:         2026
Erfolgreich getestet am: 20.08.2026
"""

import subprocess
import sys
import os
import shutil

def run_command(command):
    print(f"\n[INFO] Starte Befehl: {' '.join(command)}")
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        print(f"[FEHLER] Befehl fehlgeschlagen mit Exit-Code {process.returncode}")
        sys.exit(process.returncode)

def main():
    NEMO_MODEL_PATH = "/data/llama3_2_3b.nemo"
    TRAIN_DATA = "/data/nemo-fraud-detection/data/sft/train.jsonl"
    VAL_DATA = "/data/nemo-fraud-detection/data/sft/validation.jsonl"
    
    exp_dir = "nemo_experiments"
    if os.path.exists(exp_dir):
        print(f"[INFO] Lösche alten Experiment-Ordner '{exp_dir}'...")
        shutil.rmtree(exp_dir)

    print("\n=== Starte Finetuning (Sauber mit WandB & deaktiviertem TFLOPs-Callback) ===")
    finetune_cmd = [
        "python", 
        "/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py",
        "--config-path=/opt/NeMo/examples/nlp/language_modeling/tuning/conf",
        "--config-name=megatron_gpt_finetuning_config",
        
        # Modell
        f"model.restore_from_path={NEMO_MODEL_PATH}",
        
        # Training Dataset
        f"model.data.train_ds.file_names=['{TRAIN_DATA}']",
        "model.data.train_ds.concat_sampling_probabilities=[1.0]",
        
        # Validation Dataset
        f"model.data.validation_ds.file_names=['{VAL_DATA}']",
        
        # Hardware
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "model.tensor_model_parallel_size=1",
        "model.pipeline_model_parallel_size=1",
        
        # Training
        "trainer.max_steps=10",
        "trainer.val_check_interval=5",
        
        # --------------------------------------------------
        # Logging
        # --------------------------------------------------
        # TensorBoard aus
        "++exp_manager.create_tensorboard_logger=False",
        # W&B an
        "++exp_manager.create_wandb_logger=True",
        "++exp_manager.wandb_logger_kwargs.project=nemo-fraud-detection",
        "++exp_manager.wandb_logger_kwargs.name=llama3-2-3b-sft",
        # TFLOPs/GFLOPS Callback komplett deaktivieren
        "++exp_manager.log_tflops_per_sec_per_gpu=False",
        
        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------
        "++model.data.train_ds.ds_type=megatron",
        "++model.data.validation_ds.ds_type=megatron",
        "++model.data.train_ds.prompt_template='{input}{output}'",
        "++model.data.validation_ds.prompt_template='{input}{output}'"
    ]
    
    run_command(finetune_cmd)
    print("\n[ERFOLG] Finetuning erfolgreich abgeschlossen!")

if __name__ == "__main__":
    main()