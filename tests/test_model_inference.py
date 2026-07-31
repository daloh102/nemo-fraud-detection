import json
import pytest
from unittest.mock import MagicMock, patch

# Passe den Import an das Modul an, in dem deine Inferenz-Logik liegt
# z.B. from src.models.inference import run_inference, parse_model_output


# ==============================================================================
# DUMMY / HELPER FUNCTIONS (Falls noch nicht importiert)
# ==============================================================================
def parse_model_output(raw_response: str) -> dict:
    """Sicherheits-Parser für Modell-Antworten."""
    if not raw_response or not raw_response.strip():
        return {"label": "unknown", "confidence": 0.0}
    
    cleaned = raw_response.strip().lower()
    if "fraud" in cleaned:
        return {"label": "fraud", "confidence": 1.0}
    elif "legit" in cleaned:
        return {"label": "legit", "confidence": 1.0}
    
    return {"label": "unknown", "confidence": 0.0}


def run_inference(model_runner, text: str) -> dict:
    """Führt Inferenz aus und verarbeitet die Ausgabe."""
    if not text or not text.strip():
        raise ValueError("Eingabetext darf nicht leer sein.")
    
    prompt = f"Analyse dialogue for fraud: {text}"
    raw_response = model_runner.generate(prompt)
    
    result = parse_model_output(raw_response)
    result["prompt_length"] = len(prompt)
    return result


# ==============================================================================
# TESTS
# ==============================================================================
def test_parse_model_output_valid_labels():
    """Prüft, ob Fraud- und Legit-Antworten korrekt extrahiert werden."""
    assert parse_model_output("The call is fraud.")["label"] == "fraud"
    assert parse_model_output("This seems legit.")["label"] == "legit"
    assert parse_model_output("FRAUD_DETECTED")["label"] == "fraud"


def test_parse_model_output_fallback():
    """Prüft das Fallback-Verhalten bei unklaren Antworten oder Leerstrings."""
    assert parse_model_output("")["label"] == "unknown"
    assert parse_model_output("   ")["label"] == "unknown"
    assert parse_model_output("I am not sure about this call.")["label"] == "unknown"


def test_run_inference_success():
    """Simuliert einen erfolgreichen Inferenz-Aufruf mit einem Mock-Modell."""
    mock_model = MagicMock()
    mock_model.generate.return_value = "Result: fraud"

    input_text = "Hallo, geben Sie mir Ihre Kreditkartennummer."
    output = run_inference(mock_model, input_text)

    # Sicherstellen, dass das Modell mit dem korrekten Prompt aufgerufen wurde
    mock_model.generate.assert_called_once()
    assert "Kreditkartennummer" in mock_model.generate.call_args[0][0]
    
    # Rückgabewerte prüfen
    assert output["label"] == "fraud"
    assert output["confidence"] == 1.0


def test_run_inference_empty_input():
    """Stellt sicher, dass leere Eingaben frühzeitig mit Fehler abgefangen werden."""
    mock_model = MagicMock()

    with pytest.raises(ValueError, match="Eingabetext darf nicht leer sein"):
        run_inference(mock_model, "")

    with pytest.raises(ValueError):
        run_inference(mock_model, "   \n  ")

    # Modell darf gar nicht erst aufgerufen worden sein
    mock_model.generate.assert_not_called()


def test_run_inference_extreme_length():
    """Prüft, dass extrem lange Texte ohne Absturz verarbeitet werden."""
    mock_model = MagicMock()
    mock_model.generate.return_value = "legit"

    long_text = "Sehr langer Text. " * 5000  # ~80.000 Zeichen
    output = run_inference(mock_model, long_text)

    assert output["label"] == "legit"
    assert output["prompt_length"] > 80000