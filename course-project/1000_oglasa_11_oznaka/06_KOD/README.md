# OPJ NER: reprodukcioni repozitorijum

Ovaj repozitorijum sadrzi finalni kod, ulazne podatke i zamrznute podele za
prepoznavanje imenovanih entiteta u IT oglasima. Jedinica eksperimenta je ceo
oglas (`ad_id`): naslov i telo nikada nisu podeljeni izmedju train i test dela.

Korpus ima 1.000 oglasa i span anotacije oblika `(ad_id, field, start, end,
label)` nad izvornim UTF-8 tekstom. Svih sest modela koristi isti
`splits/outer_folds.json`: 10-fold, group-aware multilabel-stratified podelu.
Jedan ad i svaki cluster byte-identicnih body tekstova uvek ostaju u istom
foldu. Inner podele su unapred zamrznute.

## Struktura

- `collection/` — scraper i opis prikupljanja;
- `annotation/` — HTML alat, finalno uputstvo i tehnicki format izvoza;
- `data/` — sirovi oglasi, finalne anotacije, kanonski skup i kalibracija 50;
- `splits/` — jedinstveni finalni outer i inner splitovi;
- `src/common`, `src/data`, `src/splits`, `src/features` — zajednicka infrastruktura;
- `src/models/` — pokretaci za NB, SVM, XGBoost, CRF, BERTic i mBERT;
- `src/evaluation/` — span/token metrike i pooled-OOF finalizacija;
- `src/audit/` — nezavisni recomputation audit rezultata;
- `src/analysis/` — corpus, duplicate, agreement, GPT i revision analize;
- `src/report/` — tabele i grafikoni iz kanonskih pooled final rezultata.

`results/` se popunjava pokretanjem eksperimenata. Checkpoint-i, HF cache i
raniji eksperimentalni izlazi nisu ukljuceni u repozitorijum.

`configs/*.json` sadrze **referentna eksperimentalna podesavanja** radi
dokumentovanja eksperimenta. Pokretaci koriste CLI podrazumevane vrednosti
definisane uz odgovarajuci launcher.

## Instalacija i priprema

Potreban je Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python -m src.data.prepare_ner_dataset
python -m src.checks.validate_experiment_protocol
```

Priprema proverava offsete, tagove i preklapanja, zatim materijalizuje
`data/processed/ner_dataset.json` i `.jsonl`. Protokol proverava text-only
tokenizaciju, integritet 10 foldova i da duplicate clusteri nisu presečeni.

Zamrznuti split se koristi neposredno. Sledece komande su samo za regeneraciju
splita kada je to potrebno:

```powershell
python -m src.splits.make_outer_folds
python -m src.splits.make_inner_folds --inner-folds 10
python -m src.splits.make_inner_folds --inner-folds 5
```

## Pokretanje modela

### Lokalni CPU

```powershell
python -m src.models.naive_bayes.run_naive_bayes
python -m src.models.svm.run_svm
python -m src.models.crf.run_crf
```

NB koristi 10 outer × 10 inner foldova. SVM i CRF koriste 10 outer × 5 inner
foldova. Za jedan kratak run koristi se, na primer, `--folds 0`.

### Google Colab: jedan T4 GPU

U Colab-u se bira T4 GPU, repozitorijum se uploaduje/raskopuje, a iz root-a se
pokrece:

```python
!pip install -q -r requirements.txt -r requirements-colab-t4.txt
!python -m src.checks.validate_experiment_protocol
!python -m src.models.xgboost.run_xgboost
!python -m src.models.bertic.run_bertic
!python -m src.models.mbert.run_mbert
```

XGBoost, BERTic i mBERT koriste jednu T4 GPU i foldove pokrecu sekvencijalno.
Transformeri zapisuju OOF rezultat za svaku epohu 1–7, bez checkpoint-a po
default-u.

## Kanonska evaluacija: 10 OOF foldova -> final rezultat

Nakon sto svih 10 outer foldova jednog modela zavrsi, `finalize_results` cita
njegove standardne OOF span i token fajlove i pravi jedini kanonski rezultat:

```powershell
python -m src.evaluation.finalize_results --require-all
```

Time se za svaki klasicni model formiraju:

```text
results/<model>/final/metrics.json
results/<model>/final/per_label.csv
results/<model>/final/fold_metrics.csv
```

Za transformere nastaje isti skup fajlova u `final/epoch_1/` do
`final/epoch_7/`; sa `--require-all` nijedan Transformer se ne finalizuje dok
svih sedam epoha nema kompletnih deset OOF foldova. `metrics.json` je pooled micro rezultat nad svim OOF
predikcijama, a `fold_metrics.csv` je samo dopunska fold dijagnostika. Nije
prosek fold F1 vrednosti.

```powershell
python -m src.audit.recompute_metrics_from_oof --model-dir results\naive_bayes
python -m src.audit.recompute_metrics_from_oof --model-dir results\bertic --epoch 7
python -m src.report.make_model_report_artifacts
```

Audit iznova racuna metrike iz OOF fajlova i zahteva identicnost sa sacuvanim
`final/metrics.json` **i** `final/per_label.csv`. Report generator cita samo te
pooled final artefakte i generise glavnu modelsku tabelu, Transformer epoch
krivulje, per-label strict-F1 tabelu/grafikon i invalid-BIO dijagnostiku.

## Analize korpusa i anotacija

```powershell
python -m src.analysis.corpus_stats
python -m src.analysis.duplicate_audit
python -m src.analysis.human_agreement
python -m src.analysis.gpt_vs_human
python -m src.analysis.revision_diff_stats --initial data\analysis\revision_inputs\init_annotations_1000.json
```

`data/analysis/calibration_50/` sadrzi cetiri nezavisne ljudske anotacije i
GPT anotaciju istih 50 oglasa. `revision_inputs/init_annotations_1000.json`
je odvojeni INIT skup koji se koristi za analizu izmena anotacija i nije deo
finalnog 1.000-ad korpusa.

## Zastita od leakage-a

- tokenizer je deterministicki i text-only; nikada ne cita gold granice;
- vocabulary klasicnih modela fituje se samo na train delu splita;
- feature matrica se gradi direktno kao CSR;
- outer split je multilabel-stratified na nivou celog oglasa;
- svi modeli citaju isti `outer_folds.json`;
- rezultati sadrze dataset/split SHA-256, tokenizer i protocol verziju.
