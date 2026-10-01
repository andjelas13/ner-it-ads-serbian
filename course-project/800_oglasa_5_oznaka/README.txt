OPJ NER — minimalni predajni paket: 800 IT oglasa, pet oznaka

Ovaj folder sadrži materijale tražene za predaju podataka i dokumentacije,
i uz njih izvorni kod eksperimenta sa rezultatima.

1. 01_ORIGINALNI_TEKSTOVI_800/oglasi_800_utf8.jsonl.txt
   Jedan UTF-8 TXT/JSONL fajl sa 800 originalnih oglasa. Svaki red sadrži ad_id, naslov, telo, URL, izvor i source ID.

2. 02_FINALNE_ANOTACIJE_800/finalne_anotacije_800_span.tsv
   Finalne span anotacije u UTF-8 TSV formatu: ad_id, field, start, end, label, text.
   Offset je 0-based, end-exclusive: text je field[start:end]. Ukupno spanova: 5814.

3. 03_KALIBRACIJA_50_ODVOJENO
   Četiri odvojene ljudske anotacije nad istih 50 kalibracionih oglasa, filtrirane na pet oznaka.
   Broj anotacija: {'A': 442, 'B': 330, 'C': 388, 'D': 383}.

4. 04_UPUTSTVO/Uputstvo_za_anotaciju_CPU_GPU_RAM_SKLADISTE_CENA.txt
   Finalno uputstvo za pet korišćenih oznaka.

5. 05_KOD/
   Izvorni kod eksperimenta nad ovim skupom podataka.

   src/            prikupljanje, priprema skupa, podela na foldove, odlike, modeli, evaluacija, izveštajni artefakti
   scripts/        build_dataset_800.py gradi kanonski skup iz fajlova pod 01_ i 02_;
                   make_annotation_batches_800.py bira 800 oglasa od 1000 i deli ih na četiri dela po 200
   data/           kanonski skup (data/processed/ner_dataset.json i .jsonl) i ulazne anotacije
   splits/         fiksna podela na 10 spoljašnjih foldova
   configs/        konfiguracije modela
   collection/     scraper korišćen za prikupljanje oglasa
   annotation/     HTML alat korišćen za anotaciju
   requirements.txt, requirements-colab-t4.txt

   results/        OOF predikcije svih šest modela iz svih 10 spoljašnjih foldova, i finalni
                   pooled rezultati po modelu:
                     naive_bayes, svm, xgboost, crf     fold_XX_oof_predictions.json
                                                        fold_XX_oof_token_predictions.json
                                                        final/metrics.json, final/per_label.csv
                     bertic, mbert                      fold_XX/epoch_N_oof_predictions.json
                                                        fold_XX/epoch_N_oof_token_predictions.json
                                                        final/epoch_N/metrics.json, per_label.csv
                   Transformer modeli su praćeni kroz sedam epoha; poređenje u izveštaju je na 7. epohi.

   report_artifacts/
                   tables/    model_metrics.csv i .json, per_label_metrics.csv, invalid_bio_comparison.csv
                   figures/   grafikoni koje pravi src/report/make_model_report_artifacts.py
                   analysis/  statistika korpusa, provera duplikata, saglasnost anotatora

   Brojevi u results/ i report_artifacts/ dobijeni su iz istih predikcija.
   Pooled rezultat se može ponovo izračunati iz predikcija, na primer:

     python -m src.evaluation.evaluate_predictions \
       --predictions results/crf/fold_0*_oof_predictions.json \
       --token-predictions results/crf/fold_0*_oof_token_predictions.json \
       --output results/crf/final

Oznake: CPU, GPU, RAM, SKLADISTE, CENA.
Svi tekstualni fajlovi su UTF-8.
