"""Experiment-wide constants. Keep these byte-identical in both ZIP packages."""
LABELS = [
    "CPU", "GPU", "RAM", "SKLADISTE", "CENA",
]
BIO_LABELS = ["O"] + [f"{p}-{label}" for label in LABELS for p in ("B", "I")]
LABEL_TO_ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
SEED = 42
TOKEN_PATTERN = r"\w+(?:[-./+]\w+)*|[^\w\s]"

