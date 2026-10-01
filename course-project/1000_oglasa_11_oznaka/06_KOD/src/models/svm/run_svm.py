"""Run linear SVM locally on CPU with the frozen 10x5 nested-CV protocol."""
from src.models.classical_runner import run_model

if __name__ == "__main__":
    raise SystemExit(run_model("svm", 5, gpu=False))
