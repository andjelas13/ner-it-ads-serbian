# Named Entity Recognition in Serbian IT Classified Ads

Course project in Natural Language Processing, School of Electrical Engineering, University of Belgrade, 2026.
Author of this repository: Anđela Spasić.

A corpus of Serbian IT classified ads annotated with labels for computer components and price, six models trained on it, and an individual extension that compares the first automatic annotation pass with the final human annotations, case by case. The corpus and the models are the work of a team of four students, in which the author took part; the extension is her individual work.

## What the project does

The goal is to pull structured information out of ordinary classified ads: which processor, graphics card, memory and storage a seller is offering, and at what price. Ads are collected from a classifieds site, the relevant parts of the text are annotated by hand under a written guideline, and the annotated set is then used to train and evaluate six models, from Naive Bayes and CRF to BERTic and mBERT, with fixed cross validation folds so that results can be compared.

The extension turns the question around and looks at the annotations themselves. How much do two annotators agree once the guideline is settled, and where does an automatic first pass of annotation differ from what people decided in the end? Each difference is reduced to a single case, classified by its shape, and the hard ones are judged against the guideline, example by example.

## Contents

The repository has two parts.

`course-project/` is the team deliverable: the corpus, the annotation guideline and tool, the training and evaluation code, and the model results. It was made by a team of four students, including the author. The author's part of that work was the annotation of 200 of the 800 ads, the Naive Bayes model, and the presentation of results.

`individual-extension/` is the author's own work: a repeated annotator calibration and a systematic comparison of the first LLM annotation pass with the final human annotations.

## Corpus

1,000 ads were collected from a Serbian classifieds site, and 800 of them carry the final labelling scheme.

Annotations are character spans over the original text, in the form (ad_id, field, start, end, label), where the field is the title or the body of the ad. The 800 ad set has 5,814 spans.

The final scheme has five labels: CPU, GPU, RAM, SKLADISTE (storage) and CENA (price). The first round used eleven labels, including screen, battery, brand, model, warranty and location. Those six were dropped and the set was annotated again.

The ads are ordinary user texts: inconsistent spelling, missing diacritics, tables pasted from specification sheets, several products in one ad. That is what makes the decisions about span boundaries difficult.

The repository also holds the annotation guideline, the HTML annotation tool, the scraper, fixed 10 fold splits, and four independent annotations of the same 50 calibration ads.

Seller contact details are masked. Four phone numbers and one e-mail address that sellers left in the ad text were replaced with X characters of the same length, so every character offset and every annotation stays valid.

## Models in the team part

Ten fold group aware cross validation, where one ad never appears in both the training and the test part of a fold. Pooled out of fold strict F1 on the 800 ad set:

| Model | Strict F1 |
|---|---:|
| mBERT | 0.644 |
| CRF | 0.633 |
| BERTic | 0.589 |
| SVM | 0.384 |
| Naive Bayes | 0.204 |
| XGBoost | 0.010 |

Strict F1 counts a span as correct only when both boundaries and the label match. Per label tables, figures and all metric files are in `course-project/800_oglasa_5_oznaka/05_KOD/report_artifacts` and in the `results` folder next to it. Per fold prediction dumps take 167 MB and are not part of this repository; rerunning the experiments produces them again.

## Individual extension

It has three parts, all under `individual-extension/`.

The first is a repeated calibration. Two annotators independently annotated the same 50 ads again, under the final guideline with five labels. Agreement between them rose from strict F1 0.7575 to 0.8452, semantic token F1 from 0.9297 to 0.9598, and the number of structural disagreements fell from 92 to 54.

The second is the comparison of the first LLM pass with the final human annotations on all 800 ads: strict F1 0.8395, relaxed F1 0.9210, semantic token F1 0.9293, with bootstrap confidence intervals computed over ads. Differences are not counted span by span. Spans that overlap are linked into one region, and that region is one case of difference, classified by the shape of the difference: span present on one side only, wider, narrower or shifted boundary, fragmentation, merging, wrong label. This gives 1,114 cases of difference in 311 ads.

The third is a case study. A stratified sample of 100 cases was drawn, and 35 of them were analysed one by one against the guideline: what the ad says, what each side marked, which rule applies and which annotation is better. The examples are then grouped by the rule the difference turns on, which separates real model errors from places where the guideline itself gives no answer.

The report is in `individual-extension/05_IZVESTAJ` and is written in Serbian.

### Running the extension

```
pip install -r individual-extension/07_KOD/requirements.txt
cd individual-extension
python -X utf8 07_KOD/analiziraj_llm_i_rucne.py
python -X utf8 07_KOD/formiraj_uzorak_100.py
python -X utf8 07_KOD/uporedi_staru_i_novu_kalibraciju.py --andjela-letter D --nikola-letter C --new-andjela 03_NOVA_KALIBRACIJA/Andjela/nova_kalibracija_Andjela_50_oglasa_5_oznaka.json --new-nikola 03_NOVA_KALIBRACIJA/Nikola/nova_kalibracija_Nikola_50_oglasa_5_oznaka.json
python -X utf8 07_KOD/proveri_dodatne_mere_tokena_i_granica.py
python -X utf8 07_KOD/napravi_grafikone.py
```

Python 3.11 or newer is needed, with numpy, matplotlib and openpyxl. The sampling script uses a fixed seed and checks a SHA-256 fingerprint of the selection, so every run gives the same 100 cases.

## Note on the data

The ads were collected for coursework at the faculty and are kept here as the reference data behind the published numbers. Apart from the masked contact details, the texts are as they were published. The code and the annotations are the authors' own work.
