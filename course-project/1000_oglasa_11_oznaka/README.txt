OPJ NER — clean predajni paket: 1000 IT oglasa, jedanaest oznaka

01_IZVESTAJ.pdf
  Finalni projektni izveštaj.

02_UPUTSTVO_ZA_ANOTACIJU.pdf
  Finalno anotaciono uputstvo korišćeno u projektu.

03_OGLASI/
  1000 originalnih UTF-8 TXT oglasa i metadata.tsv sa izvorom, URL-om i jedinstvenim ID-jem.

04_ANOTACIJE/
  Finalne anotacije glavnog skupa u span-TXT i BIO/CoNLL formatu.
  Podfolder kalibracija sadrži tekstove/metapodatke za 50 kalibracionih oglasa i četiri zasebne ljudske anotacije A-D u istom span-TXT formatu.

05_ALATI/
  Scraper korišćen za prikupljanje podataka i HTML alat korišćen za anotaciju.

06_KOD/
  Izvorni kod eksperimenta nad ovim skupom podataka.

  src/            prikupljanje, priprema skupa, podela na foldove, odlike, modeli, evaluacija, izveštajni artefakti
  data/           kanonski skup i ulazne anotacije
  splits/         fiksna podela na 10 spoljašnjih foldova
  configs/        konfiguracije modela
  collection/     scraper korišćen za prikupljanje oglasa
  annotation/     HTML alat korišćen za anotaciju
  requirements.txt, requirements-colab-t4.txt

  report_artifacts/analysis/
                  saglasnost anotatora, poređenje automatske predanotacije sa ljudskom,
                  i pregled izmena u fazi revizije

  Rezultati modela za ovu verziju nalaze se u 01_IZVESTAJ.pdf; same OOF predikcije
  nisu čuvane u repozitorijumu, pa je results/ prazan.

Oznake: CPU, GPU, RAM, SKLADISTE, EKRAN, BATERIJA, BRAND, MODEL, GARANCIJA, CENA, MESTO.
Svi tekstualni fajlovi su UTF-8.
