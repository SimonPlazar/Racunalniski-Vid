# Detektor ključnih točk

Pri tej nalogi boste implementirali in naučili nevronsko mrežo za detekcijo ključnih točk po članku SuperPoint: Self-Supervised Interest Point Detection and Description. Članek opisuje arhitekturo in učenje nevronske mreže za detekcijo in opis ključnih točk (podobno kot metode ORB, SIFT, FAST in druge). V sklopu naloge bi pripravili in učili samo detekcijo točk.

V okviru naloge boste pripraviti sintetične slike za učenje, sestavili in naučili nevronsko mrežo, uporabili tehniko homografske adaptacije za izboljšanje rezultatov.

V sklopu te naloge:

priprava generatorja sintetičnih učnih slik z ustreznim ciljem 30 točk

posamezni trikotniki, štirikotniki, zvezde 10 točk
šahovnice, kocke, nabor likov v isti sliki 10 točk
uporaba homografije 10 točk
priprava in učenje nevronske mrež 60 točk

priprava arhitekture 20 točk
priprava učnih primerov 20 točk
učenje 10 točk
diagram izgube med učenjem 10 točk
implementacija homografske adaptacije za napoved 25 točk

demonstracija delovanja na fotografiji 10 točk

Na sistem oddajte spisano kodo in pripravljeno poročilo (PDF format). Tukaj imate primer poročila za to nalogo.

Poročilo
V poročilu prikažite:

primere sintetično generiranih učnih slik z označenimi točkami
diagram ali izpis nevronske mreže
graf izgube skozi učenje (najbolje iz tensorboard orodja)
primere rezultatov na sintetičnih slikah
primere rezultatov na resničnih fotografijah
primere rezultatov z uporabe homografske adaptacije
Priprava sintetičnih slik
Za nadzorovano učenje potrebujemo slike na katerih so naši cilji označeni. Za označevanje ključnih točk na fotografijah je problem sam nejasno opredeljen. Vemo kakšne točke želimo: takšne, katerim lahko natančno določimo lokacijo in podamo dober opis. Vendar ne vemo katere točke v sliki ustrezajo našim željam. Idealno bi problem definirali tako, da bi se mreža lahko učila nenadzorovano - sama bi poiskala takšne točke, ki imajo jasno določljivo pozicijo in jih je preprosto povezati med slikami (takšne mreže obstajajo, vendar so precej kompleksne in zahtevne za učenje).

Kot preprostejša alternativa lahko sami generiramo učne vzorce. V izvirnem članku predlagajo generiranje slik s preprostimi liki: trikotniki, štirikotniki, zvezdami, ... (primeri so na sliki 4 v članku). Točke teh likov nam nato služijo kot ciljne ključne točke.

Za generiranje boste potrebovali knjižnico za risanje. Za te primere potrebujete metode za risanje črt in zapolnjenih poligonov. Uporabite lahko OpenCV ali sciki-image knjižnico.

Za vse primere naključno spreminjate barvo ozadja in barvo likov. Poskrbeti morate le, da bo razlika dovolj velika. Če sta barva ozadja in lika preveč podobna je to slab učni primer. V slike dodaste tudi nekaj zglajenega gausovega šuma. Zaradi arhitekture nevronske mreže morate poskrbeti, da bo ločljivost slik deljiva z 8.

Preprostejši liki
Za tri in štirikotnike je generiranje preprosto. Izberemo 3 ali 4 naključne točke v sliki in izrišemo lik. Naključno izbrane točke so tudi ciljne ključne točke.

Za generiranje zvezde izberemo naključno središčno točko in nato 5 naključnih zunanjih točk okrog. Razdalje med točkami morajo presegati nekakšno minimalno mejo. Vsako zunanjo točko povežemo s središčno. Vseh 6 točk je ciljnih ključnih točk.

Zahtevnejši liki
Za generiranje šahovnice v sliko izrišemo šahovnico in naključno obarvamo njena polja. Ključne točke so stičišča različnih polj. Za šahovnico 3x4 polj, imamo 20 ciljnih ključnih točk: 4*5 točk, oziroma 6 v notranjosti in 14 po zunanjem robu. Za dodatne transformacije si pomagamo s homografijo (opisano spodaj).

Za generiranje 3D kocke lahko uporabimo podoben postopek kot za zvezdo. Izberemo 1 središčno točko, okrog te generiramo 3 notranje robne točke, ki jih povežemo z središčno. Tako smo ustvarili središčni del. Sedaj dodamo še 3 zunanje robne točke, ki povezujejo notranje. S temi točkami sedaj izrišemo 3 štirikotnike in dobimo projekcijo 3D kocke. Vseh 7 točk je označenih ključnih točk.

Za generiranje večih preprostih likov znotraj ene slike moramo biti pozorni, da se le ti ne prekrivajo ali ustvarijo novih kotičkov, ki ne bi bili označeni. Za vsak lik lahko posebej preverimo, ali se njegov izris prekriva z že obstoječimi. Če se prekriva ga zavržemo in poskusimo novega.

Uporaba homografije za dodatno augmentacijo primerov
Primere lahko dodatno augmentiramo (in si tako tudi poenostavimo zgornje generiranje) z uporabo naključnih perspektivnih/homografskih transformacij. Žal ne moremo samo naključno generirati matrike homografske transformacije, ker vse možne transformacije ne dajejo smiselnih rezultatov. Vsebino generirane slike lahko povsem pomaknejo izven vidnega območja slike ali pa neprepoznavno zvijejo. V članku zato priporočajo sestavljanje homografije iz skrbno zbranih osnovnih transformacij (slika 6 in poglavje 5.2 v članku).

Namesto tega lahko uberemo majhno bljižnico. V sliki naključno izberemo 4 točke, vsako v svojem predelu slike: levo zgoraj, desno zgoraj, desno spodaj, levo spodaj. Paziti moramo samo, da izbrane točke niso preblizu središča slike.

Za te 4 točke nato izračunamo homografsko transformacijo, ki jih preslika v 4 kotičke slike.

Za naključno rotacijo, lahko pred izračunom homografije še naključno premaknemo za 1-4 mesta. Tako se bo rezultat še naključno rotiral za 90 stopinj.

Transformacijo uporabimo za augmentacijo učne slike in označenih točk. V tej točki moramo paziti, da označene točke, ki padejo izven mej slike odstranimo iz ciljnih primerov.

Enak postopek augmentacije slik uporabimo za homografsko adaptacijo.

Priprava nevronske mreže
Predlagana nevronska mreža (slika 3 v članku) je sestavljena iz enkoderja (poglavje 3.1) in dveh dekoderjev. Nas zanima samo dekoder za detekcijo ključnih točk (poglavje 3.2). Podrobnosti slojev so podane v poglavju 6.

Enkoder
Za kodiranje značilk pripravite naslednjo nevronsko mrežo (enaka arhitektura kot v prvi nalogi):

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
Dekoder
Izhod kodirnika je slika katere višina in širina je 1/8 originalne. Vsak piksel ima 128 vrednosti - to je vektor značilnic, ki sedaj opisuje 8x8 celico originalne slike. Dekoder za detekcijo točk to sliko poda skozi naslednje sloje:

konvolucija, 3x3 velikost, 256 filtrov
"batch normalization"
ReLU
konvolucija, 1x1 velikost, 65 filtrov
Izhod na tem mestu je slika velikosti H/8 in W/8 s 65 kanali. Ta slika je v članku označena s pisanim X (na sliki 3 in formulah 1, 2, 3).

Teh 65 kanalov vsakega piksla predstavlja razrede ene izmed 64 lokacij v 8x8 celici originalne slike, ter en dodaten razred za primer, kadar v celici ni bilo prisotne točke. Ta izhod uporabimo kot izhodni nivo učne mreže, za katerega moramo pripraviti ustrezne učne primere.

Za končni izhod mreže to sliko X podamo še skozi softmax sloj, odrežemo zadnji kanal, ter preoblikujemo to sliko v velikost originalne slike. To lahko storimo z uporabo tensorflow metode depth_to_space, ali pa osnovnih slojev Reshape in Permute v keras (reshape in transpose v numpy). Z osnovnimi koraki je postopek naslednji:

Reshape v velikost H/8, W/8, 8, 8
Permute kanalov v vrstni red H/8, 8, W/8, 8
Reshape v velikost H, W
Uporaba osnovnih slojev je morda jasnejša, ampak potencialno počasnejša od depth_to_space (tensorflow) ali PixelShuffle (pytorch).

Za napoved točk sedaj v tej sliki poiščemo lokalne maksimume, ki so večji od nekega izbranega pragu.

Priprava učnih primerov je obraten postopek. Če pripravimo ciljno masko, v kateri so pozicije točk označene z 1 ostali piksli pa imajo vrednost 0. Potem takšno masko pretvorimo v učni vzorec po korakih:

Reshape v velikost H/8, 8, W/8, 8
Permute kanalov v vrstni red H/8, W/8, 8, 8
Reshape v velikost H/8, W/8, 64
dodamo kanal 65
za vsak izmed H/8, 8, W/8 pikslov preverimo, da je nastavljena ena izmed 65 točk
kjer je nastavljenih več ohranimo eno naključno (poglavje 3.4, opomba 2)
kjer ni nastavljena nobena nastavimo kanal 65
Izguba in učenje
Za učenje uporabimo izgubo opisano v formulah 2 in 3 (uporabimo samo izgubo za detekcijo točk, za opis ignoriramo).

Formula 2 opisuje samo povprečje preko vseh celic. Če je izhod mreže več vzorcev (kot v našem primeru manjša slika), je to privzeto obnašanje.

Formula 3 opisuje izgubo izračunano v posameznem pikslu - celici izhodne slike. Formula opisuje kategorično križno entropijo v nekoliko drugačni obliki.

Znotraj logaritma imamo softmax funkcijo preko kanalov slike X v posamezni celici. Sam izračun odvoda izgube je lahko natančnejši/bolj stabilen, kadar je softmax upoštevan pri izračunu. Zato kot učni izhod funkcije ne uporabimo izhoda softmax sloja uporabljene zgoraj, ampak kot učni izhod uporabimo vhod v softmax funkcijo. Ta vhod pogosto imenujemo logit. To moramo ustrezno označiti tudi v funkciji izgube, ki jo uporabimo.

V imenovalcu vidimo da formula 3 uporabi samo tisti kanal, ki ustreza označbi v podatkih. To je samo okrajšava splošnega zapisa križne entropije. Le ta bi seštela preko vseh kanalov, vendar logaritme pomnožila z oznakami. Tako bo ostala samo tista, katere vrednost je 1, ostale pa so pomnožene z 0 in se izgubijo.

Dodatne podrobnosti učenja so navedene v poglavju 6:

vhodne slike so sivinske v velikosti 240x320 (HxW)
učenje z sintetičnimi slikami poteka 200,000 iteracij
velikost učnega paketa je 32 slik (batch size)
za optimizacijo je uporabljen algoritem ADAM, z učnim korakom 0.001 in parametrom Beta = 0.9, 0.999
Velikost učnega paketa, velikost slik in število iteracij prilagodite svoji strojni opremi. Če bo problem zasedel preveč rama, zmanjšajte velikost učnega paketa ali velikost slik. Če bo učenje prepočasno zmanjšajte število iteracij (ali pa ga zgodaj prekinite). Pričakovan čas učenja je 8-10 ur (v članku je bilo 20 ur s pomočjo grafično pospešenega računanja).

Homografska adaptacija
Mreža naučena na takšnih sintetičnih podatkih ne daje stabilnih rezultatov (primer v sliki 7). Rezultate lahko izboljšamo z uporabo homografske adaptacije (poglavje 5).

Za napovedovanje točk na sliki, na vhodno sliko apliciramo naključne homografske transformacije (kot so opisane zgoraj v generiranju sintetičnih primerov). Slike podamo skozi nevronsko mrežo, da pridobimo rezultate. Rezultate poravnamo nazaj z inverzom homografskih transformacij in jih povprečimo. V tej povprečni sliki detekcij sedaj poiščemo lokalne maksimume. Za dobre rezultate uporabimo 99 naključnih homografij + originalno sliko.

Učenje mreže v obeh primerih lahko traja dalj časa in zasede večje količino pomnilnika. Parametre učenja prilagodite svoji strojni opremi, ki vam je na voljo. Če zmanjšate ločljivosti slik in velikost učnega paketa (batch) bo učenje potekalo hitreje z manj pomnilnika. Če učenje traja predolgo (>12 ur) lahko zmanjšate število učnih korakov. Naučen model lahko sproti shranjujete. Tako lahko kadarkoli prekinete učenje in ga nadaljujete kasneje. Ali obnovite rezultate ob nepričakovanih prekinitvah.