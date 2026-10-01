"""Run Naive Bayes locally on CPU with the frozen 10x10 nested-CV protocol."""
from src.models.classical_runner import run_model

# 10 unutrašnjih foldova; sama ugnežđena validacija je u src/classical/nested_cv.py
if __name__ == "__main__":
    raise SystemExit(run_model("nb", 10, gpu=False))
