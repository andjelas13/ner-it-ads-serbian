"""Run BERTiÄ‡ on a Google Colab T4 GPU using the shared frozen outer split."""
from src.models.transformer_runner import run_model

if __name__ == "__main__":
    raise SystemExit(run_model("bertic"))
