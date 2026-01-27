# Ocenjevanje homografije

V sklopu te naloge:

generirajte učne primere 25 točk
za regresijo 15 točk
za klasifikacijo 10 točk
pripravite in učite nevronsko mrežo 50 točk
resnet blok 10 točk
regresijska glava 10 točk
klasifikacijska glava 20 točk
diagrami izgube med učenjem 10 točk
ovrednotite naučeno mrežo 50 točk
ovrednotite ocenjevanje naučenih mrež 10 točk
primerjate rezultate s klasično metodo ocene homografije 40 točk
Na sistem oddajte spisano kodo in pripravljeno poročilo v PDF formatu.

Tukaj imate primer poročila za to nalogo.

Generiranje učnih slik
Generiranje slik je opisano v sekciji 3 izbranega članka, prikazano v sliki 3.

Pri ocenjevanju homografije rešujemo problem, kjer isto sceno fotografiramo pod različnimi zornimi koti z zelo specifičnimi premiki kamere. Zaradi tega je zelo preprosto generirati večje količine učnih primerov brez ročnega označevanja učnih podatkov.

Za osnovo učnih slik potrebujemo zbirko naravnih fotografij. Uporabili bi lahko katerokoli zbirko, v izvirnem članku uporabijo MS CoCo zbirko. Potrebovali bomo približno 100 fotografij.

Učne slike generiramo po naslednjih korakih:

izberemo naključno sliko
sliko prevzorčimo na izbrano velikost (320x240 pikslov) in pretvorimo v sivinsko (to lahko naredimo vnaprej za celotno zbirko)
v sliki izberemo okno velikost 64x64 na naključni lokaciji (pazimo da ne izbiramo preblizu roba)
4 kotičke okna pomaknemo za naključne pomike v intervalu [-16, 16]
iz 4 kotičkov in njihovih pomikov izračunamo homografijo H
inverz ocenjene homografije H^-1 apliciramo na sliko, dobimo transformirano sliko
iz slike in transformirane slike izrežemo vzorca z izbranim oknom
vzorca zložimo v sliko z dvema kanaloma, to je vhod v mrežo
Za učenje lahko slike pripravljate sproti ali pa vnaprej pripravite večjo zbirko vzorcev.

Nevronska mreža
Za nevronsko mrežo pripravite naslednjo arhitekturo (se zgleduje po izvirnem članku, nekoliko manjša):

2 ResNet bloka
64 kanalov, 3x3 filtri
"batch normalization" sloj
ReLU prenosna funkcija
max pooling sloj
2 ResNet bloka
64 kanalov, 3x3 filtri
"batch normalization" sloj
ReLU prenosna funkcija
max pooling sloj
2 ResNet bloka
128 kanalov, 3x3 filtri
"batch normalization" sloj
ReLU prenosna funkcija
max pooling sloj
2 ResNet bloka
128 kanalov, 3x3 filtri
"batch normalization" sloj
ReLU prenosna funkcija
polno povezan sloj
512 kanalov
Na vrh mreže nato dodate eno izmed dveh vrst glave.

Regresijska glava:

polno povezan sloj
8 izhodov
učimo z evklidsko izgubo (L2, RMSE)
Klasifikacijska glava:

polno povezan sloj
8*21 izhodov
preoblikovanje
v 8, 21
softmax sloj
preko 21 vrednosti
učimo z izgubom križne entropije
Učenje
Naučiti morate dve mreži. Prva z regresijsko glavo in druga z klasifikacijsko glavo. Za učenje uporabite ustrezno izgubo za vsako glavo.

Adam optimizator s hitrostjo učenja približno 1e-4 (to je lahko odvisno od mnogih podrobnosti implementacije). Učimo lahko 50.000 korakov z 1 vzorcem na korak.

Za generiranje učnih vzorcev lahko uporabimo tudi samo 1 sliko.

Za učenje na CPU lahko dobre rezultate dosežemo v manj kot eni uri učenja (preizkušeno na AMD Ryzen 7 4700U, 8 jeder, 2-4 GHz).

Preizkus pripravljene mreže
Preden poskusite pripravljeno mrežo učiti na večji zbirki naključno pripravljenih vzorcev je dobro preveriti, da je pripravljena arhitektura pravilno povezana in pripravljena.

Osnovni preizkus pripravljene mreže in parametrov učenja lahko opravite z učenjem na 1 vzorcu. Če je mreža pravilno povezana, če je ciljna vrednost pravilno oblikovana in če so parametri učenja smotrni bi učenje moralo precej hitro konvergirati k skoraj identičnem rezultatu.

Takšen preizkus ulovi nekatere nekatere nadležne napake, ki se nam zgodijo v pripravi mreže (napačno povezani sloji, napačno izbrana funkcija izgube).

Diagram izgube in napake med učenjem
Pričakovano je, da boste v poročilu dodali diagrame spremembe izgube tekom učenja (izguba v odvisnosti od koraka).

Takšne diagrame lahko pridobite z uporabo orodji za vizualizacijo napredka učenja. Lokalno je široko sprejeto orodje (TensorBoard), katerega preprosto uporabimo v TensorFlow ali PyTorch knjižnici. Podobne diagrame pa nam pripravijo tudi spletne storitve kot so Weights and Biases.

Seveda lahko diagram izrišete tudi sami po učenju z uporabo knjižnice kot je matplotlib.

Evalvacija
Naučeno mrežo lahko poskusimo ovrednotiti na zbirki slik, ki niso bile uporabljene pri učenju. Testne primere generirate z enakim postopkom kot učne primere.

Uporabite približno 100 slik, za vsako sliko v tej generirajte 10 testnih primerov (skupaj približno 1.000 testnih primerov).

Rezultate ovrednotite s korenom srednje kvadratične napake (RMSE) ocenjenih zamikov kotičkov okna. Narišite diagram napak iz katerega bo razviden raztros napak (histogram, boxplot) različnih metod in poročajte smiselno povzeto napako (povprečje ali mediana, standardni odklon).

Za poročilo pripravite tudi nekaj primerov poravnave slik z ocenjeno homografijo.

Primerjava s klasičnim pristopom
Za primerjavo metode enako ovrednotite tudi ocenjevanje homografije z detektorji točk v knjižnici OpenCv (SIFT, SURF ali ORB). Implementacije najdete v podanih primerih zgoraj.

Da bo primerjava smiselna boste morali napako izračunati na enak način kot zgoraj - RMSE zamikov kotičkov okna.

Lahko se zgodi da s klasičnim pristopom ne uspete oceniti homografije, običajno ker metoda morda ne detektira dovolj točk. V takšnih primerih najbolje predpostaviti, da je ocenjena homografija identiteta.

Klasični pristop bo delal zelo slabo na vzorcih 64x64 pikslov. Da bo imel primerljive pogoje, mu dajte vzorce velikosti 256x256 pikslov z zamiki kotičkov 64 pikslov, ocenjene napake pa nato delite z 4.
