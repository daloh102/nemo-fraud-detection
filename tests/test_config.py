import os
import pytest
import yaml
from pathlib import Path

# Basisverzeichnis des Projekts bestimmen
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "conf"  # Falls deine Configs in einem anderen Ordner liegen (z.B. src/conf), hier anpassen!


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def find_all_yaml_files():
    """Sucht alle .yaml und .yml Dateien im Projektverzeichnis."""
    if not CONFIG_DIR.exists():
        # Fallback auf BASE_DIR, falls kein separater conf/-Ordner existiert
        return list(BASE_DIR.glob("**/*.yaml")) + list(BASE_DIR.glob("**/*.yml"))
    return list(CONFIG_DIR.glob("**/*.yaml")) + list(CONFIG_DIR.glob("**/*.yml"))


# ==============================================================================
# TESTS
# ==============================================================================
def test_config_directory_exists():
    """Prüft, ob überhaupt YAML-Konfigurationsdateien existieren."""
    yaml_files = find_all_yaml_files()
    assert len(yaml_files) > 0, f"Keine YAML-Konfigurationsdateien in {BASE_DIR} gefunden!"


@pytest.mark.parametrize("yaml_path", find_all_yaml_files())
def test_yaml_syntax_validity(yaml_path):
    """
    Parametrisierter Test: Lädt jede YAML-Datei einzeln und prüft, 
    ob sie syntaktisch korrektes YAML enthält.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            assert config is not None, f"Die Konfigurationsdatei '{yaml_path.name}' ist leer!"
            assert isinstance(config, dict), f"Die Wurzel von '{yaml_path.name}' muss ein Dictionary sein!"
        except yaml.YAMLError as exc:
            pytest.fail(f"Syntaxfehler in YAML-Datei '{yaml_path}':\n{exc}")


def test_sft_or_model_config_structure():
    """
    Sucht nach Haupt-Konfigurationsdateien (z. B. sft_config.yaml oder model_config.yaml)
    und prüft kritische Schlüssel und Wertschnitte.
    """
    yaml_files = find_all_yaml_files()
    
    # Sucht nach typischen Konfigurationsdateien
    model_configs = [p for p in yaml_files if "sft" in p.name.lower() or "model" in p.name.lower() or "train" in p.name.lower()]
    
    if not model_configs:
        pytest.skip("Keine spezifische Modell-/SFT-Config-Datei für Struktur-Check gefunden.")

    for cfg_path in model_configs:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Prüfe gängige Trainingseinstellungen, falls vorhanden
        if "trainer" in cfg or "training" in cfg:
            trainer_cfg = cfg.get("trainer") or cfg.get("training")
            
            # Max Epochs / Steps Check
            if "max_epochs" in trainer_cfg:
                assert isinstance(trainer_cfg["max_epochs"], int)
                assert trainer_cfg["max_epochs"] > 0, "max_epochs muss größer als 0 sein."
                
            # Batch Size Check
            if "batch_size" in trainer_cfg:
                assert isinstance(trainer_cfg["batch_size"], int)
                assert trainer_cfg["batch_size"] > 0, "batch_size muss positiv sein."

        # Learning Rate Check (falls in config vorhanden)
        if "learning_rate" in cfg:
            lr = cfg["learning_rate"]
            assert isinstance(lr, (float, str)), "Learning Rate muss eine Zahl/String sein."
            assert float(lr) < 1.0, "Learning Rate erscheint unüblich hoch (>= 1.0)!"


def test_path_references_in_configs():
    """
    Überprüft, ob in den Configs angegebene Pfad-Strings keine ungültigen Zeichen
    enthalten und korrekt formatiert sind.
    """
    yaml_files = find_all_yaml_files()
    
    for cfg_path in yaml_files:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        if not isinstance(cfg, dict):
            continue

        # Rekursive Suche nach Keys, die 'path', 'dir' oder 'file' heißen
        def check_paths(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    check_paths(v)
                elif isinstance(k, str) and ("path" in k.lower() or "dir" in k.lower()):
                    if isinstance(v, str) and v.startswith("/"):
                        # Prüft, ob absolute Pfade nicht versehentlich Windows-Slashes nutzen
                        assert "\\" not in v, f"Ungültiger Pfad-Separator in '{k}': {v}"

        check_paths(cfg)