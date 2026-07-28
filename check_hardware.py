import subprocess
import shutil
import sys
import os

class HardwareCheckError(Exception):
    """Eigene Exception für Hardware-/Umgebungsfehler."""
    pass

def check_disk_space(required_gb=300):
    """Prüft, ob genügend Festplattenspeicher für Container und Modell-Cache vorhanden ist."""
    print("🔍 [1/4] Prüfe freien Festplattenspeicher...")
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    
    print(f"   --> Freier Speicherplatz: {free_gb:.2f} GB")
    
    if free_gb < required_gb:
        raise HardwareCheckError(
            f"Zu wenig Speicherplatz! Benötigt: mind. {required_gb} GB, Verfügbar: {free_gb:.2f} GB.\n"
            f"Tipp: Das Mistral-128B Modell und das NeMo Container-Image benötigen viel Platz."
        )
    print("   ✅ Speicherkapazität ausreichend!\n")

def check_nvidia_smi():
    """Prüft, ob nvidia-smi verfügbar ist und liest die verfügbaren GPUs aus."""
    print("🔍 [2/4] Prüfe Host-NVIDIA-Treiber & GPU-Verfügbarkeit...")
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = result.stdout.strip().split("\n")
        print(f"   --> Gefundene GPUs ({len(gpus)}):")
        
        total_free_vram = 0
        for gpu in gpus:
            idx, name, mem_total, mem_free = [x.strip() for x in gpu.split(",")]
            free_vram = int(mem_free) / 1024
            total_free_vram += free_vram
            print(f"       • GPU {idx}: {name} | Freier VRAM: {free_vram:.2f} GB / {int(mem_total)/1024:.2f} GB")
            
        print(f"   --> Gesamter freier VRAM über alle GPUs: {total_free_vram:.2f} GB")
        
        if total_free_vram < 80:
            print("   ⚠️ WARNUNG: Der freie VRAM könnte für ein 128B-Modell knapp werden!")
        
        print("   ✅ Host GPU-Treiber erreichbar!\n")
        
    except FileNotFoundError:
        raise HardwareCheckError("'nvidia-smi' wurde nicht gefunden. Bitte installiere die NVIDIA-Treiber auf dem Host.")
    except subprocess.CalledProcessError as e:
        raise HardwareCheckError(f"Fehler bei der Ausführung von nvidia-smi: {e.stderr}")

def check_docker_installation():
    """Prüft, ob Docker installiert ist und der Docker-Daemon läuft."""
    print("🔍 [3/4] Prüfe Docker-Installation & Daemon-Status...")
    
    try:
        subprocess.run(["docker", "info"], capture_output=True, text=True, check=True)
        print("   ✅ Docker ist installiert und der Daemon läuft!\n")
    except FileNotFoundError:
        raise HardwareCheckError("Docker CLI nicht gefunden. Ist Docker auf dem System installiert?")
    except subprocess.CalledProcessError:
        raise HardwareCheckError("Docker-Daemon läuft nicht oder der aktuelle Benutzer hat keine Rechte (sudo / docker group).")

def check_nvidia_container_toolkit():
    """Testet, ob Docker Zugriff auf die GPUs hat (NVIDIA Container Runtime Test)."""
    print("🔍 [4/4] Prüfe NVIDIA Container Toolkit (GPU-Passthrough in Docker)...")
    
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "nvidia/cuda:12.0.0-base-ubuntu22.04",
        "nvidia-smi"
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("   ✅ GPU-Durchreichung an Docker-Container funktioniert einwandfrei!\n")
    except subprocess.CalledProcessError as e:
        raise HardwareCheckError(
            f"Docker kann nicht auf die GPUs zugreifen! Fehler:\n{e.stderr}\n"
            "Tipp: Stelle sicher, dass das 'nvidia-container-toolkit' installiert ist und Docker neu gestartet wurde."
        )

def run_all_checks():
    print("==================================================================")
    print("🚀 STARTE HARDWARE- UND ENVIRONMENT-CHECK VOR DEM CONTAINER-START")
    print("==================================================================\n")
    
    try:
        check_disk_space(required_gb=250)  # Schwellenwert in GB (anpassbar)
        check_nvidia_smi()
        check_docker_installation()
        check_nvidia_container_toolkit()
        
        print("==================================================================")
        print("🎉 ALLES PERFEKT! Deine Hardware ist bereit für Docker Compose.")
        print("==================================================================")
        return True
        
    except HardwareCheckError as e:
        print("\n❌ HARDWARE CHECK FEHLGESCHLAGEN!")
        print("------------------------------------------------------------------")
        print(f"Fehlerdetails: {e}")
        print("------------------------------------------------------------------")
        print("Bitte behebe das obige Problem, bevor du 'docker compose up' startest.")
        return False

if __name__ == "__main__":
    success = run_all_checks()
    if not success:
        sys.exit(1)