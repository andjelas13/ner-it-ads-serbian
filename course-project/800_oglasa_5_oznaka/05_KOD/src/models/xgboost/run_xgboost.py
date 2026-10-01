"""Run XGBoost sequentially on one Google Colab T4 GPU (frozen 10x10 CV)."""
from src.models.classical_runner import run_model

if __name__ == "__main__":
    raise SystemExit(run_model("xgb", 10, gpu=True))
