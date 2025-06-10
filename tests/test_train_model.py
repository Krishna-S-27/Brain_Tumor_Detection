from machine_learning.src.train_model import train_model

def test_train_model_runs():
    metrics = train_model()
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.7  # Example threshold
