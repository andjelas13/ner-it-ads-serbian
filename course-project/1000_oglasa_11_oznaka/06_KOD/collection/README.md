# Kako je prikupljen skup od 1.000 oglasa

Oglasi su prikupljeni restartabilnom Python skriptom `scraper.py` sa sajtova
Halo Oglasi i KupujemProdajem. Finalna raspodela je 700 Halo Oglasi i 300
KupujemProdajem oglasa, uz unapred zadate kvote po IT kategorijama: laptop,
telefon, tablet, desktop racunar, monitor, GPU, RAM, skladiste, CPU, baterija,
maticna ploca i napajanje.

Za svaki prihvacen oglas cuvaju se naslov, telo/opis, izvor, interna kategorija,
kategorija izvora, URL, `source_ad_id` i vreme preuzimanja. Skripta odbacuje
neaktivne oglase, stranice van ocekivane IT kategorije, oglase za kupovinu ili
otkup, zapise bez naslova/tela i opise krace od osam karaktera. Za preiroke
kategorije postoje jednostavni tekstualni uslovi u samoj skripti, npr. da oglas
za bateriju zaista sadrzi izraz povezan sa baterijom.

Kandidati se mesaju deterministicki seed-om `20260824`, a zatim se uzimaju do
popunjavanja tacnih kvota. Kalibracionih 50 oglasa se iskljucuju po identitetu
izvornog oglasa. Validacija proverava tacno 1.000 zapisa, obavezna polja,
dozvoljene izvore i kategorije, kvote i jedinstvenost kanonskog URL/ID kljuca.

Nisu primenjivani dodatni kriterijumi tezine, duzine ili ocekivanog broja NER
anotacija. Drugim recima, oglasi nisu naknadno birani da budu "laki" ili "teski"
za anotiranje.

`scraper.py` sadrzi implementaciju postupka prikupljanja podataka. Pri pokretanju
generise manifest i kratak izvestaj o raspodeli; ti izvedeni fajlovi nisu deo
zamrznutog repozitorijuma.
