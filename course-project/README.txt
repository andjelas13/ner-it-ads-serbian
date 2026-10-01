OPJ — Prepoznavanje imenovanih entiteta u oglasima za IT opremu

Projekat ima dve verzije eksperimenta. Druga je dopuna prve, ne zamena — obe se
predaju, jer se rezultati porede.

1000_oglasa_11_oznaka/
  Prva verzija. 1.000 oglasa, 11 oznaka (23 BIO klase), 14.321 spanova.
  Anotacija: automatska predanotacija jezičkim modelom, pa ljudska revizija.
  Najbolji model: CRF, pooled strict F1 0,6675.

800_oglasa_5_oznaka/
  Druga verzija. 800 oglasa izabranih iz istog korpusa, 5 oznaka (11 BIO klasa),
  5.814 spanova. Anotacija: potpuno ručna, po 200 oglasa po članu tima.
  Najbolji model: mBERT posle 7. epohe, pooled strict F1 0,6437.

Šta se između verzija nije menjalo: podela na 10 spoljašnjih foldova, ugnežđena
unakrsna validacija, tokenizator, tri metrike (strict, relaxed-overlap,
semantic-token), zaštite od curenja podataka i skup od šest modela
(Naive Bayes, SVM, XGBoost, CRF, BERTić, mBERT).

Šta se menjalo: faza anotacije je ponovljena bez pomoći jezičkog modela, a šema
je smanjena sa 11 na 5 oznaka.

Svaki folder ima svoj README.txt sa detaljnim sadržajem.
Svi tekstualni fajlovi su UTF-8.
