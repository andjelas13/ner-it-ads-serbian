"""GPU entry point for restart-safe seven-epoch Transformer CV."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.classical.models import make_sequences
from src.common.constants import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID, SEED
from src.data.prepare_ner_dataset import entity_set, examples, load_json
from src.evaluation.metrics import full_report
from src.common.protocol import result_matches, run_identity
from src.splits.make_outer_folds import load_fold
from src.common.tokenization import invalid_bio_diagnostics
from src.transformers.transformer_data import (
    TorchChunkDataset,
    build_chunks,
    chunks_to_entities,
    chunks_to_word_labels,
    collator,
    model_name,
    subword_length_stats,
)


DEFAULT_EPOCHS = 7
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/ner_dataset.json"
DEFAULT_SPLITS = PROJECT_ROOT / "splits/outer_folds.json"


def _log(message: str) -> None:
    print(message, flush=True)


def _ads(ads: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    return [ad for ad in ads if str(ad["id"]) in ids]


def _trim_batch_predictions(
    predictions: np.ndarray,
    chunk_indices: Sequence[int],
    chunks: Sequence[Any],
) -> list[np.ndarray]:
    """Remove dynamic batch padding and preserve the original chunk order."""
    return [
        np.asarray(row[: len(chunks[int(chunk_index)].input_ids)], dtype=np.int64)
        for row, chunk_index in zip(predictions, chunk_indices)
    ]


def _amp_optimizer_step_succeeded(old_scale: float, new_scale: float, use_amp: bool) -> bool:
    """GradScaler lowers its scale when an inf/nan makes it skip optimizer.step()."""
    return not use_amp or new_scale >= old_scale


def _evaluate(
    model: Any,
    dataset: TorchChunkDataset,
    chunks: list[Any],
    sequences: list[Any],
    gold: set,
    device: Any,
    batch_size: int = 16,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    predicted_by_chunk: list[np.ndarray | None] = [None] * len(chunks)
    with torch.no_grad():
        for batch_number, batch in enumerate(
            DataLoader(dataset, batch_size=batch_size, collate_fn=collator), 1
        ):
            chunk_indices = [int(value) for value in batch["chunk_index"].tolist()]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            padded_predictions = (
                model(input_ids=input_ids, attention_mask=attention_mask)
                .logits.argmax(-1).cpu().numpy()
            )
            trimmed = _trim_batch_predictions(padded_predictions, chunk_indices, chunks)
            for chunk_index, row in zip(chunk_indices, trimmed):
                predicted_by_chunk[chunk_index] = row
            if batch_number % 50 == 0:
                _log(f"  evaluation batch {batch_number}")

    missing = [index for index, row in enumerate(predicted_by_chunk) if row is None]
    if missing:
        raise RuntimeError(f"Evaluation produced no prediction for chunk indices: {missing[:20]}")
    predicted_rows = [row for row in predicted_by_chunk if row is not None]
    word_labels = chunks_to_word_labels(sequences, chunks, predicted_rows)
    predicted = chunks_to_entities(sequences, chunks, predicted_rows)
    metrics = full_report(gold, predicted, [sequence.labels for sequence in sequences], word_labels)
    metrics["invalid_bio_before_repair"] = invalid_bio_diagnostics(word_labels)
    return {"metrics": metrics, "predictions": sorted(predicted), "predicted_bio": word_labels}


def run_fold(
    alias: str,
    fold: int,
    data_path: Path,
    split_path: Path,
    results: Path,
    max_length: int | None = None,
    stride: int = 32,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    checkpoint_policy: str = "none",
    log_every: int = 50,
) -> None:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    start = time.perf_counter()
    name = model_name(alias)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = results / alias / f"fold_{fold:02d}"
    completion_file = output / f"epoch_{epochs}_metrics.json"
    if (result_matches(completion_file, data_path, split_path, alias, fold, epochs)
            and (output / f"epoch_{epochs}_oof_predictions.json").exists()
            and (output / f"epoch_{epochs}_oof_token_predictions.json").exists()):
        _log(f"SKIP model={alias} fold={fold}: matching completed protocol already exists")
        return
    output.mkdir(parents=True, exist_ok=True)

    _log(f"START model={alias} fold={fold} epochs={epochs} device={device}")
    _log("loading and validating dataset")
    data = load_json(data_path)
    all_ads = examples(data)
    train_ids, test_ids = load_fold(split_path, fold)
    train_ads, test_ads = _ads(all_ads, train_ids), _ads(all_ads, test_ids)
    train_sequences, test_sequences = make_sequences(train_ads), make_sequences(test_ads)

    _log(f"loading tokenizer: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    _log("computing train-only subword length statistics")
    stats = subword_length_stats(train_sequences, tokenizer)
    chosen_length = max_length or int(stats["recommended_max_length"])
    _log(
        f"building chunks max_length={chosen_length} stride={stride} "
        f"train_sequences={len(train_sequences)} test_sequences={len(test_sequences)}"
    )
    train_chunks = build_chunks(train_sequences, tokenizer, chosen_length, stride)
    test_chunks = build_chunks(test_sequences, tokenizer, chosen_length, stride)
    _log(f"chunks ready: train={len(train_chunks)} test={len(test_chunks)}")
    train_ds, test_ds = TorchChunkDataset(train_chunks), TorchChunkDataset(test_chunks)

    _log(f"loading model: {name}")
    model = AutoModelForTokenClassification.from_pretrained(
        name,
        num_labels=len(BIO_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    optimiser = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(loader) / gradient_accumulation_steps)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_updates * 0.1),
        num_training_steps=total_updates,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    gold = entity_set({"examples": test_ads})
    (output / "length_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    optimizer_updates = 0
    skipped_amp_updates = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        _log(f"epoch {epoch}/{epochs}: training ({len(loader)} batches)")
        model.train()
        optimiser.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "chunk_index"
            }
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model(**batch).loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % gradient_accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = float(scaler.get_scale())
                scaler.step(optimiser)
                scaler.update()
                new_scale = float(scaler.get_scale())
                if _amp_optimizer_step_succeeded(old_scale, new_scale, use_amp):
                    scheduler.step()
                    optimizer_updates += 1
                else:
                    skipped_amp_updates += 1
                optimiser.zero_grad(set_to_none=True)
            if step % log_every == 0 or step == len(loader):
                _log(
                    f"  epoch {epoch}/{epochs} step {step}/{len(loader)} "
                    f"loss={float(loss.detach().cpu()) * gradient_accumulation_steps:.6f}"
                )

        _log(f"epoch {epoch}/{epochs}: evaluating")
        evaluated = _evaluate(
            model, test_ds, test_chunks, test_sequences, gold, device
        )
        predictions = evaluated.pop("predictions")
        predicted_bio = evaluated.pop("predicted_bio")
        (output / f"epoch_{epoch}_oof_predictions.json").write_text(
            json.dumps(
                [
                    {"ad_id": ad, "field": field, "start": start_offset,
                     "end": end_offset, "label": label}
                    for ad, field, start_offset, end_offset, label in predictions
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        token_oof = [
            {"ad_id": sequence.ad_id, "field": sequence.field,
             "tokens": [{"start": token.start, "end": token.end} for token in sequence.tokens],
             "predicted_bio": labels}
            for sequence, labels in zip(test_sequences, predicted_bio)
        ]
        (output / f"epoch_{epoch}_oof_token_predictions.json").write_text(
            json.dumps(token_oof, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        checkpoint_saved: str | None = None
        should_save = checkpoint_policy == "all" or (
            checkpoint_policy == "last" and epoch == epochs
        )
        if should_save:
            checkpoint = output / f"checkpoint_epoch_{epoch}"
            _log(f"saving checkpoint: {checkpoint}")
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            checkpoint_saved = str(checkpoint)

        payload = {
            "model": alias,
            **run_identity(data_path, split_path),
            "pretrained_checkpoint": name,
            "checkpoint_saved": checkpoint_saved,
            "checkpoint_policy": checkpoint_policy,
            "outer_fold": fold,
            "epoch": epoch,
            "configured_epochs": epochs,
            "metrics": evaluated["metrics"],
            "max_length": chosen_length,
            "stride": stride,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": 2e-5,
            "optimizer_updates_so_far": optimizer_updates,
            "skipped_amp_updates_so_far": skipped_amp_updates,
            "epoch_runtime_seconds": time.perf_counter() - epoch_start,
            "runtime_seconds_so_far": time.perf_counter() - start,
            "device": str(device),
            "python": sys.version,
            "platform": platform.platform(),
        }
        (output / f"epoch_{epoch}_metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log(
            f"epoch {epoch}/{epochs}: done strict_f1={evaluated['metrics']['f1']:.6f} "
            f"minutes={(time.perf_counter() - epoch_start) / 60:.1f}"
        )

    _log(
        f"END model={alias} fold={fold} epochs={epochs} "
        f"minutes={(time.perf_counter() - start) / 60:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bertic", "mbert"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--results", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--checkpoint-policy", choices=["none", "last", "all"], default="none"
    )
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    run_fold(
        args.model,
        args.fold,
        args.data,
        args.splits,
        args.results,
        args.max_length,
        args.stride,
        args.epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.checkpoint_policy,
        args.log_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
