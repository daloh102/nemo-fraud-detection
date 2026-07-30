import pytest
# Passe den Import an dein Evaluator-Modul an
# from src.evaluator.metrics import calculate_metrics 

def calculate_metrics_dummy(y_true, y_pred):
    """Beispiel-Implementierung der Metrik-Berechnung für den Test."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "fraud" and p == "fraud")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "legitimate" and p == "fraud")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "fraud" and p == "legitimate")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "legitimate" and p == "legitimate")

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def test_perfect_metrics():
    y_true = ["fraud", "legitimate", "fraud", "legitimate"]
    y_pred = ["fraud", "legitimate", "fraud", "legitimate"]
    
    metrics = calculate_metrics_dummy(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_zero_division_handling():
    # Modell sagt nur 'legitimate' voraus
    y_true = ["fraud", "fraud"]
    y_pred = ["legitimate", "legitimate"]
    
    metrics = calculate_metrics_dummy(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0