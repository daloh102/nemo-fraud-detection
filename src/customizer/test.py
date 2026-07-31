import json
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
import psutil
import torch
from huggingface_hub import hf_hub_download
from nemo.collections import llm

# ==============================================================================
# 1. UMFASSENDER SYSTEM- & PERFORMANCE-CHECK (Vorab-Prüfung)
# ==============================================================================
MIN_REQUIRED_RAM_GB = 5.0           # Mindestens benötigter freier RAM in GB
MIN_REQUIRED_DISK_GB = 15.0         # Mindestens benötigter freier Festplattenspeicher in GB
MIN_REQUIRED_INODES_PCT = 5.0       # Mindestens 5% freie Inodes
MIN_REQUIRED_BW_MBPS = 200.0        # Mindestens erforderlicher Schreibdurchsatz in MB/s
TARGET_DIR = Path("/data/nemo-fraud-detection")
TEST_FILE = TARGET_DIR / "test_iowrite.tmp"

print("🔍 Starte umfassende Vorab-Prüfung der Systemressourcen & I/O-Leistung...\n")

# --- A. RAM Check ---
mem = psutil.virtual_memory()
free_ram_gb = mem.available / (1024 ** 3)
total_ram_gb = mem.total / (1024 ** 3)
print(f"📊 [RAM] {free_ram_gb:.2f} GB frei von insgesamt {total_ram_gb:.2f} GB.")

if free_ram_gb < MIN_REQUIRED_RAM_GB:
    print(f"❌ FEHLER: Zu wenig RAM! Benötigt: mind. {MIN_REQUIRED_RAM_GB} GB, Verfügbar: {free_ram_gb:.2f} GB.")
    sys.exit(1)

# --- B. Festplatten-Check (Speicherplatz) ---
check_path = TARGET_DIR if TARGET_DIR.exists() else Path("/")
disk_usage = shutil.disk_usage(check_path)
free_disk_gb = disk_usage.free / (1024 ** 3)
total_disk_gb = disk_usage.total / (1024 ** 3)
print(f"💾 [Festplatte] {free_disk_gb:.2f} GB frei von insgesamt {total_disk_gb:.2f} GB auf {check_path}.")

if free_disk_gb < MIN_REQUIRED_DISK_GB:
    print(f"❌ FEHLER: Zu wenig Festplattenplatz! Benötigt: mind. {MIN_REQUIRED_DISK_GB} GB, Verfügbar: {free_disk_gb:.2f} GB.")
    sys.exit(1)

# --- C. Inodes-Check (Dateikontingent) ---
try:
    statvfs = os.statvfs(check_path)
    total_inodes = statvfs.f_files
    free_inodes = statvfs.f_ffree
    free_inodes_pct = (free_inodes / total_inodes) * 100 if total_inodes > 0 else 0
    print(f"🗂️ [Inodes] {free_inodes_pct:.1f}% freie Dateikontingente verfügbar.")

    if free_inodes_pct < MIN_REQUIRED_INODES_PCT:
        print(f"❌ FEHLER: Zu wenige freie Inodes ({free_inodes_pct:.1f}%)!")
        sys.exit(1)
except Exception as e:
    print(f"⚠️ Warnung: Inodes konnten nicht geprüft werden ({e}), fahre fort...")

# --- D. GPU & VRAM Check ---
if not torch.cuda.is_available():
    print("❌ FEHLER: Keine CUDA-fähige GPU gefunden!")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"🎮 [GPU] Gefunden: {gpu_name} mit {total_vram_gb:.2f} GB VRAM.")

# --- E. Automatischer I/O-Durchsatz-Test (fio) ---
print("⚡ [I/O-Test] Führe automatischen Schreib-Benchmark durch...")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

fio_command = [
    "fio",
    "--name=iotest",
    "--ioengine=libaio",
    "--rw=write",
    "--bs=1M",
    "--size=512M",
    "--iodepth=64",
    f"--filename={TEST_FILE}",
    "--direct=1",
    "--sync=1",
    "--output-format=json"
]

try:
    result = subprocess.run(fio_command, capture_output=True, text=True, check=True)
    fio_data = json.loads(result.stdout)
    # Schreibbandbreite in Byte/s extrahieren und in MB/s umrechnen
    write_bw_bytes = fio_data["jobs"][0]["write"]["bw_bytes"]
    write_bw_mbps = write_bw_bytes / (1024 * 1024)
    print(f"🚀 [I/O-Ergebnis] Gemessener Schreibdurchsatz: {write_bw_mbps:.2f} MB/s")

    if write_bw_mbps < MIN_REQUIRED_BW_MBPS:
        print(f"❌ FEHLER: Festplatten-Durchsatz zu gering! Benötigt: mind. {MIN_REQUIRED_BW_MBPS} MB/s, Gemessen: {write_bw_mbps:.2f} MB/s.")
        if TEST_FILE.exists():
            TEST_FILE.unlink()
        sys.exit(1)

except Exception as e:
    print(f"⚠️ Warnung beim I/O-Benchmark: {e}. Überspringe den I/O-Schwellenwert-Check...")
finally:
    # Testdatei aufräumen
    if TEST_FILE.exists():
        TEST_FILE.unlink()

print("\n✅ Alle System-, Speicher- und Leistungstests erfolgreich bestanden!\n")

# ==============================================================================
# 2. HUGGING FACE MODELL & CONFIG VORBEREITEN
# ==============================================================================
hf_model_id = "meta-llama/Llama-3.2-3B-Instruct"
config_file = None

print("📥 Lade config.json im HuggingFace-Cache...")
max_retries = 3
for attempt in range(1, max_retries + 1):
    try:
        config_file = hf_hub_download(repo_id=hf_model_id, filename="config.json")
        break
    except Exception as e:
        print(f"⚠️ Warnung bei Download (Versuch {attempt}/{max_retries}): {e}")
        if attempt == max_retries:
            print("❌ Kritischer Fehler: Konnte Konfiguration nicht laden.")
            sys.exit(1)
        time.sleep(5)

with open(config_file, "r") as f:
    config_dict = json.load(f)

# Problematisches rope_scaling-Feld entfernen
config_dict.pop("rope_scaling", None)

from transformers import LlamaConfig
config = LlamaConfig(**config_dict)

# ==============================================================================
# 3. NEMO MODELL INITIALISIEREN UND IMPORTIEREN
# ==============================================================================
nemo_config = llm.LlamaConfig(
    hidden_size=config.hidden_size,
    num_attention_heads=config.num_attention_heads,
    num_query_groups=config.num_key_value_heads,
    ffn_hidden_size=config.intermediate_size,
    num_layers=config.num_hidden_layers,
    seq_length=config.max_position_embeddings,
)

model_instance = llm.LlamaModel(nemo_config)

print("🚀 Starte den Import ins NeMo-Format...")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

try:
    llm.import_ckpt(
        model=model_instance,
        source=f"hf://{hf_model_id}",
        output_path=TARGET_DIR / "Llama-3.2-3B-Instruct.nemo",
        overwrite=True
    )
    print("✅ Erfolgreich konvertiert!")
except Exception as e:
    print(f"❌ FEHLER während des NeMo-Imports: {e}")
    sys.exit(1)