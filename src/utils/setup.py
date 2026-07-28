# src/utils/setup.py
import os
import subprocess
import sys


def setup_environment(ngc_key: str = None) -> None:
    """Setzt den NGC API Key in der .env-Datei."""
    key = ngc_key or os.getenv("NGC_API_KEY")
    if not key or key == "DEIN_NVIDIA_NGC_API_KEY_HIER":
        raise ValueError(
            "❌ NGC_API_KEY fehlt! Bitte in der .env-Datei eintragen oder als Parameter übergeben."
        )

    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"NGC_API_KEY={key}\n")
    print("✅ .env-Datei aktualisiert!")


def login_docker_registry() -> None:
    """Führt den Login bei der NVIDIA Registry durch."""
    ngc_key = os.getenv("NGC_API_KEY")
    if not ngc_key:
        raise RuntimeError("❌ NGC_API_KEY ist nicht in den Umgebungsvariablen geladen.")

    print("🔒 Docker Registry Login (nvcr.io)...")
    cmd = ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "-p", ngc_key]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Docker Registry Login erfolgreich!")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Docker Login fehlgeschlagen:\n{e.stderr.strip()}") from e
    except FileNotFoundError:
        raise RuntimeError("❌ Docker CLI ist nicht auf diesem System installiert.")


def run_hardware_check(script_path: str = "check_hardware.py") -> None:
    """Führt das Hardware-Check-Skript aus."""
    print("\n🖥️ Starte Hardware-Check...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print("✅ Hardware-Check erfolgreich abgeschlossen!")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("❌ Hardware-Check mit Fehlern abgebrochen.") from e
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Skript {script_path} wurde nicht gefunden.")