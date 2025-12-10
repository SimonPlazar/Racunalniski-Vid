# Optični pretok

Pri tej nalogi boste implementirali in naučili nevronsko mrežo za napovedovanje optičnega pretoka po članku FlowNet: Learning Optical Flow with Convolutional Networks. Predpisana mreža se zgleduje po preprosti arhitekturi FlowNetSimple.

V sklopu te naloge:

pripravite nalaganje in vizualizacijo podatkov zbirke Flying chairs 15 točk
nalaganje optičnega pretoka 5 točk
vizualizacija optičnega pretoka, po zgledu članka 10 točk
pripravite in učite nevronsko mrežo 50 točk
mreža učena z samo zadnjim izhodom 20 točk
mreža učena z večnivojskimi izhodi 20 točk
diagrami izgube med učenjem (učna in validacijska zbirka) 10 točk
ovrednotite nevronsko mrežo na zbirki Sintel 60 točk
izračunana statistika 20 točk
izračunana statistika metode Farneback v OpenCV 20 točk
preizkus na lastnem posnetku, subjektivna ocena 20 točk
Na sistem oddajte spisano kodo in pripravljeno poročilo (PDF format). Tukaj imate primer poročila za to nalogo.

Učni podatki
Za učenje in validacijo uporabite zbirko podatkov "Flying chairs", ki jo prenesete tukaj.

Vizualizacija optičnega pretoka
Za vizualizacijo optičnega pretoka posnemajte prikaz v članku.

Optični pretok iz x,y koordinat pretvorite v polarne koordinate: smer in dolžino.

Smer optičnega pretoka prikažite z odtenkom barve (Hue), moč pa s saturacijo (Saturation). Svetlost nastavite po občutku, da dobite dobre rezultate (načeloma maksimalno, morda dobite boljši prikaz če nastavite nekoliko manj).

Priprava nevronske mreže
Nevronska mreža FlowNetSimple gradi na U-Net arhitekturi. V članku je arhitektura deljena na kodirnik in dekodirnik (encoder, decoder).

Vhod v mrežo je konkateniran par barvnih slik. To sta dve zaporedni sliki iz posnetka.

Izhod mreže je slika enake velikost z dvema kanaloma. Vsak kanal napoveduje optični pretok v x in y smeri. Mreža lahko ima več izhodov, eno napoved za vsak ločljivostni nivo. Te uporabimo pri učenju, pri napovedi pa nas načeloma zanima samo najvišji nivo.

Vaš kodirnik sestavite po naslednjih korakih:

Down blok (16, 7x7) → Down blok (32, 5x5) → Down blok (64, 3x3)

Conv2d (128, 3x3) → BatchNorm2d → ReLU

Conv2d (128, 3x3) → BatchNorm2d → ReLU

Down blok pa sestavite iz slojev:

Conv2d → BatchNorm2d → ReLU

Conv2d → BatchNorm2d → ReLU

MaxPool2d (2x2, 2x2)

Dekodirnik pa ima nekoliko drugačno zgradbo kot običajna U-Net mreža. Sestavite ga kot:

ConvTranspose2d (64, 2x2)
Concatenate
ConvTranspose2d (32, 2x2)
Concatenate
ConvTranspose2d (16, 2x2)
Concatenate
Sloj transponirane konvolucije poveča ločljivost vhodnih slik značilk in zmanjša število kanalov. Sloj konkatenacije pa združi to povečane slike značilk spodnjega sloja in izhod zadnjega ReLU sloja (vhod v MaxPool2d sloj) v zgornjem sloju.

Na najnižjem ločljivostnem nivoju dekodirnika na konec dodamo tudi konvolucijski sloj, ki naredi predikcijo:

Conv2d (2, 1x1)
Izhod te konvolucije želimo, da ostane takšen kot je. Ne dodajamo prenosnih funkcij kot so ReLU ali Softmax.

To napoved uporabimo pri učenju. Prenesemo pa jo tudi na višji ločljivostni nivo s pomočjo transponirane konvolucije:

ConvTranspose2d (16, 2x2)
Concatenate
Conv2d (2, 1x1)
To ponovimo na vseh ločljivostnih nivojih, dokler se ne vrnemo nazaj na originalno ločljivost.

Vizualna predstavitev arhitekture
Posamezen pravokotnik predstavlja večkanalno sliko značilnic. Nad pravokotnikom
je zapisano število kanalov in pod njem velikost (širina in višina) slike.

Legenda oznak: 
```
>    Conv2d (velikost filtra 7x7, 5x5 ali 3x3), BatchNorm2d, ReLU  
>>   Conv2d (velikost filtra 1x1)  
\\   MaxPool2d (velikost okna 2x2, premik okna 2x2)  
//   ConvTranspose2d (velikost okna 2x2, premik okna 2x2)  
--...--> Prenos vrednosti in konkatenacija  
  
6      16     16                                           (16 + 16)   32                           (32+2)  34       2  
 _     __     __                                                       ___                                  ___      _    
| |   |  |   |  |                                                     |   |                                |   |    | |   
| |   |  |   |  |                                                     |   |                                |   |    | |   
| |   |  |   |  |                                                     |   |                                |   |    | |   
| | > |  | > |  | --------------------------------------------------> |   | -----------------------------> |   | >> | |   
| |   |  |   |  |                                                     |   |                                |   |    | |   
| |   |  |   |  |                                                     |   |                                |   |    | |   
| |   |  |   |  |                                                     |   |                                |   |    | |   
|_|   |__|   |__|                                                     |___|                                |___|    |_|   
256   256    256                                                      256                                  256      256  
             \\                                                      //                                   //  
             16     32      32                           (32 + 32)   64              (64 + 2)  130        2  
              __     ___     ___                                     ____                       ____       _  
             |  |   |   |   |   |                                   |    |                     |    |     | |  
             |  |   |   |   |   |                                   |    |                     |    |     | |  
             |  | > |   | > |   | --------------------------------> |    | ------------------> |    |  >> | |  
             |  |   |   |   |   |                                   |    |                     |    |     | |  
             |  |   |   |   |   |                                   |    |                     |    |     | |  
             |__|   |___|   |___|                                   |____|                     |____|     |_|  
             128    128     128                                     128                        128        128  
                            \\                                     //                         //  
                            32       64       64        (64 + 64) 128   (128 + 2) 130         2  
                             ___     ____     ____                 _____           _____       _   
                            |   |   |    |   |    |               |     |         |     |     | |   
                            |   | > |    | > |    | ------------> |     | ------> |     |  >> | |   
                            |   |   |    |   |    |               |     |         |     |     | |    
                            |___|   |____|   |____|               |_____|         |_____|     |_|  
                            64      64       64                   64              64          64  
                                             \\                  //              //  
                                             64       128       128             2  
                                              ____     _____     _____           _    
                                             |    | > |     | > |     |   >>    | |   
                                             |____|   |_____|   |_____|         |_|   
                                             32       32        32              32      
```
Učenje nevronske mreže
Pri učenju nevronske mreže uporabite priporočano EPE napako - evklidsko razdaljo med napovedanim in pričakovanim optičnim pretokom. To napako agregirajte preko napovedi različnih ločljivostnih nivojev (seštevek ali povprečje).

Za učenje poskusite uporabiti nastavitve, ki se zgledujejo po nastavitvah v članku:

optimizacijska metoda Adam
velikost učnega paketa (batch size) 8
hitrost učenja 10^-4
število učnih korakov 500.000
Nasveti za učenje
Za hitrejše učenje na CPU-ju:

uporabimo podatkovni tip double (float64), float32 se lahko obesi
zmanjšamo velikost paketa na 1
zmanjšamo velikost učnega vzorca na 32x32
uporabimo manjše število kanalov (začnemo z 8 kanali na prvem nivoju)
Za boljše rezultate:

povečamo globino mreže (gremo na 4 nivoje)
povečamo število kanalov (začnemo z 64 kanali na prvem nivoju)
postopoma manjšamo hitrost učenja, razpolovimo vsakih 100.000 korakov od 200.000 naprej
Testiranje nevronske mreže
Za ovrednotenje nevronske mreže uporabite zbirko MPI Sintel tukaj. Poročajte povprečno napako na učni množici (za katero so podani pravilni optični pretoki).

Na enak način ovrednotite metodo za oceno gostega optičnega pretoka implementirano v OpenCV: algoritem Gunnar Farneback. Poiščite parametre pri katerih bo metoda dajala dobre rezultate.

Da tudi sami subjektivno ovrednotite rezultate pripravite kratek posnetek (dolg nekaj sekund) v katerem bo prisotno nekaj gibanja. Posnetek obdelajte z vašo naučeno nevronsko mrežo in vizualizirajte rezultate. Pripravite kratek posnetek (prikaz samega video posnetka kot njegovega optičnega pretoka), ki ga prikažete med zagovorom.