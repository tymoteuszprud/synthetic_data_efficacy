# Klasyfikacja z wykorzystaniem danych syntetycznych

Repozytorium zawiera kod eksperymentów dotyczących wykorzystania danych syntetycznych w klasyfikacji obrazów sportowych i wiadomości HAM/SPAM. Obejmuje przygotowanie danych, uczenie modeli ResNet-18 i Qwen, dodatkowe losowanie klas oraz analizy wyników.

## 1. Struktura repozytorium

| Katalog | Zawartość |
|---|---|
| `generowanie` | Przygotowanie referencji i generowanie danych |
| `obrazy` | Kolejka eksperymentów i aplikacja treningowa ResNet-18 |
| `wiadomosci` | Podział danych, trening Qwena i losowanie klas |
| `analizy` | Metryki klasyfikacyjne, liczebności i zróżnicowanie danych |
| `konfiguracje` | Parametry eksperymentów i przepływ ComfyUI |
| `instrukcje` | Instrukcje tekstowe przekazywane modelom generującym |
| `materialy` | Słownictwo pomocnicze do generowania wiadomości |
| `wymagania`, `srodowiska` | Zależności i wersje bibliotek |
| `tests` | Testy wykonywane bez modeli i bez GPU |

## 2. Środowisko wykonawcze

Polecenia należy wykonywać z głównego katalogu repozytorium. Eksperymenty obrazowe i tekstowe wymagają odrębnych środowisk Python. Wersje bibliotek wykorzystywanych podczas badań zapisano w katalogu `srodowiska`.

Manifest środowiska obrazowego wskazuje Python 3.11, a tekstowego Python 3.12. Pliki wymagań określają wersje pakietów do instalacji; manifesty zapisują wersje środowisk, z których pochodzi kod. Wariant pakietów PyTorch z obsługą CUDA wymaga instalacji zgodnej ze sterownikiem i wersją CUDA. Same pliki wymagań nie wybierają indeksu pakietów CUDA.

Środowisko podstawowe, przeznaczone do kontroli danych i analiz plików CSV, można przygotować następująco:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r wymagania/podstawowe.txt
```

Eksperymenty obrazowe wymagają zależności z pliku `wymagania/obrazy.txt`, w tym PyTorch i torchvision w wersjach zgodnych z CUDA. Eksperymenty tekstowe wymagają środowiska Unsloth oraz wersji bibliotek wskazanych w `wymagania/tekst.txt`. Instalację Unsloth opisano w [dokumentacji projektu](https://unsloth.ai/docs/get-started/install). Przygotowanie wykresów wymaga dodatkowo zależności z `wymagania/wykresy.txt`.

Generowanie danych korzysta z odrębnie uruchamianych usług [LM Studio](https://lmstudio.ai/docs) i [ComfyUI](https://docs.comfy.org/). Biblioteka Tkinter, wymagana przez interfejsy generatorów, jest częścią standardowej instalacji Python dla systemu Windows.

## 3. Dane wejściowe

Zbiory danych i wagi modeli są przechowywane lokalnie, poza repozytorium. Przykładowa struktura katalogów ma następującą postać:

```text
dane/
  tekst/
    wiadomosci_rzeczywiste.csv
    wiadomosci_syntetyczne.csv
    wiadomosci_referencyjne.csv
    indeksy_referencji.csv
  obrazy/
    obrazy_rzeczywiste/
      archery/
      ...
    obrazy_syntetyczne/
      archery/
      ...
```

Ścieżki poszczególnych źródeł obrazowych określa pole `data_sources[].path` w pliku `konfiguracje/obrazy.json`. Każdy katalog źródłowy zawiera podkatalogi klas o zgodnych nazwach. Wykaz 30 dyscyplin i ich liczebności znajduje się w `konfiguracje/klasy_obrazow.csv`.

Warianty obrazowe A, C, D i E korzystają ze wspólnego źródła `obrazy_syntetyczne`. Różnią się sposobem wykorzystania i liczebnością próbek, bez tworzenia odrębnego zbioru dla E. Przypisanie źródeł oraz ustawienia augmentacji i grup współczynnika uczenia są zapisane osobno dla każdej konfiguracji.

Pliki wiadomości mają format CSV, kodowanie UTF-8 oraz kolumny `label` i `message`. Dopuszczalne etykiety to HAM i SPAM. Plik `indeksy_referencji.csv` zawiera kolumnę `df_index` z indeksami numerowanymi od zera. Indeksy odnoszą się do kolejności wierszy w pliku danych rzeczywistych. Kopię indeksów referencyjnych zamieszczono w katalogu `konfiguracje`.

### Źródła danych i modeli

| Materiał | Źródło |
|---|---|
| Obrazy rzeczywiste | [100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification) |
| Wiadomości rzeczywiste | [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| Klasyfikator tekstowy | [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) |
| Model opisujący obrazy | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Model generujący opisy i wiadomości | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Generator obrazów | [Z-Image](https://github.com/Tongyi-MAI/Z-Image) |
| Adapter generatora obrazów | [Realistic Snapshot](https://civitai.com/models/2268008/realistic-snapshot-z-image-turbo-krea-2?modelVersionId=2617751) |
| Słownik angielski JSON | [dwyl/english-words](https://github.com/dwyl/english-words), [plik źródłowy](https://raw.githubusercontent.com/dwyl/english-words/master/words_dictionary.json) |
| Lista popularnych wyrazów — źródło uzupełniające | [powerlanguage/word-lists](https://github.com/powerlanguage/word-lists/blob/master/1000-most-common-words.txt) |

Warunki wykorzystania materiałów określają ich wydawcy. Plik przepływu ComfyUI zawiera konfigurację i nazwy modeli, bez ich wag.

### Słownictwo pomocnicze

Plik [slownictwo_podstawowe.txt](materialy/slownictwo_podstawowe.txt) jest dołączony do repozytorium i zawiera po jednym wyrazie w wierszu. Pełny słownik `materialy/slownik_angielski.json` ma postać obiektu JSON, którego kluczami są wyrazy. Jego pobranie umożliwia polecenie:

```powershell
python -m generowanie.pobierz_slownik --download
```

Skrypt sprawdza format danych i zapisuje adres źródła oraz sumę kontrolną pobranego pliku. Generator SMS wyszukuje materiały w katalogu roboczym i katalogu `materialy`. Słowa przekazywane modelowi są losowane z pełnego słownika JSON; lista podstawowa stanowi materiał pomocniczy i nie jest wymagana do uruchomienia generatora.

## Wybór klas obrazowych

W pliku `konfiguracje/wybrane_klasy.txt` umieszczane są nazwy folderów, po jednej w wierszu. Domyślnie zawiera on 30 klas pracy. Liczba klas wynika z listy, nie ze stałej w modelu. Wszystkie moduły obrazowe korzystają z tej listy albo z manifestu utworzonego na jej podstawie. Plik CSV `klasy_obrazow.csv` pozostaje historycznym zestawieniem liczebności; bieżące liczebności wynikają z plików w wybranych folderach.

Podgląd wyboru i folderów niewybranych:

```powershell
python -m obrazy.wybor_klas --root dane/obrazy/obrazy_rzeczywiste
```

Usuwanie wymaga argumentu `--remove-others` oraz wpisania wyświetlonego potwierdzenia `USUN N`. Operacja usuwa całe niewybrane podfoldery wskazanego katalogu zbioru, wraz z zawartością. Bez potwierdzenia nic nie jest usuwane. Katalog syntetyczny można uporządkować analogicznym poleceniem z inną wartością `--root`. Analizy pomijają niewybrane klasy także bez fizycznego usuwania folderów.

```powershell
python -m obrazy.wybor_klas --root dane/obrazy/obrazy_rzeczywiste --remove-others
```

Po zmianie listy tworzony jest nowy manifest; istniejący manifest innego wyboru nie jest nadpisywany. Do analiz zgodnych z konkretnym treningiem służy argument `--manifest dane/obrazy/manifest_obrazow.json`. Alternatywnie `--class-file` wskazuje własny plik nazw. Budżety rzeczywiste liczone są od pełnej wybranej puli (100% oznacza pozostałą pulę treningową); syntetyczne od rzeczywistej puli syntetycznej. W C i E poziom 100% danych syntetycznych odpowiada liczebności rzeczywistej puli treningowej.

## 4. Generowanie danych

### Wiadomości

Przygotowanie zbalansowanego podzbioru referencyjnego o liczebności odpowiadającej 1% pełnego zbioru odbywa się za pomocą polecenia:

```powershell
python -m generowanie.referencje_sms --input dane/tekst/wiadomosci_rzeczywiste.csv --output-dir dane/tekst
```

Zapisywane są pliki `wiadomosci_referencyjne.csv` i `indeksy_referencji.csv`. Istniejące pliki nie są nadpisywane.

Po przygotowaniu słownika i uruchomieniu modelu generującego w LM Studio interfejs generatora jest dostępny przez polecenie:

```powershell
python -m generowanie.uruchom sms --workspace dane/tekst
```

W formularzu określa się identyfikator modelu dostępnego na serwerze, docelową liczbę wiadomości i plik wyjściowy. Domyślna liczebność wynosi 6000. Wiadomości są dopisywane do pliku `wiadomosci_syntetyczne.csv`. Instrukcje generowania zamieszczono w `instrukcje/ham.txt` i `instrukcje/spam.txt`.

### Obrazy

Najpierw przygotowywany jest wspólny manifest klas, referencji i podziałów dla wszystkich ziaren. Domyślnie wybierane są nazwy z `konfiguracje/wybrane_klasy.txt`. Argument `--classes` przyjmuje listę wybranych nazw; nazwy zawierające spacje wymagają cudzysłowów, np. `"canoe slalom"`. Argument `--class-file` pozwala wskazać inny wykaz. Liczebności budżetów wyznaczane są z aktualnej liczby plików wybranych klas. Referencje są wyłączone z walidacji i testu dla każdego ziarna. Manifest jest współdzielony przez opisywanie, generowanie i trening.

```powershell
python -m obrazy.dane --real dane/obrazy/obrazy_rzeczywiste --output dane/obrazy/manifest_obrazow.json
```

Etap opisywania wymaga modelu Qwen3.5-9B uruchomionego w LM Studio. Lista referencji pochodzi z manifestu.

```powershell
python -m generowanie.opisy opisz --manifest dane/obrazy/manifest_obrazow.json --images dane/obrazy/obrazy_rzeczywiste --output dane/generowanie_obrazow/referencje.txt --generate
```

Rozwinięcie opisów scen wykorzystuje model Qwen3.6-35B-A3B. Parametr `--fraction` określa liczbę przygotowywanych opisów względem liczebności klasy; wartość domyślna wynosi 0,25.

```powershell
python -m generowanie.opisy rozwin --manifest dane/obrazy/manifest_obrazow.json --images dane/obrazy/obrazy_rzeczywiste --input dane/generowanie_obrazow/referencje.txt --output dane/generowanie_obrazow/opisy_scen.txt --generate
```

Plik `konfiguracje/przeplyw_generowania_obrazow.json` należy umieścić w katalogu roboczym generatora. Model Z-Image Turbo, enkoder tekstu, VAE i adapter Realistic Snapshot muszą być dostępne w ComfyUI pod nazwami określonymi w konfiguracji przepływu.

```powershell
Copy-Item konfiguracje/przeplyw_generowania_obrazow.json dane/generowanie_obrazow/
$env:COMFY_OUTPUT_DIR = 'D:/ComfyUI/output'
python -m generowanie.uruchom obrazy --manifest dane/obrazy/manifest_obrazow.json --workspace dane/generowanie_obrazow --output dane/obrazy/obrazy_syntetyczne
```

Generowanie odbywa się w rozdzielczości 512 × 512 pikseli, w partiach po cztery obrazy, z wykorzystaniem sześciu kroków i adaptera o wadze 0,6. Wyniki są zapisywane w podkatalogach klas.

Ostatnia partia klasy może być mniejsza, aby nie przekroczyć docelowej liczebności. Generowanie wymaga przygotowanych opisów scen; brak pliku lub opisów danej klasy nie jest zastępowany ogólną instrukcją.

## 5. Eksperymenty tekstowe

Warianty uczenia obejmują:

- **A:** uczenie na całym zbiorze syntetycznym, następnie dostrajanie na danych rzeczywistych;
- **B:** uczenie wyłącznie na danych rzeczywistych;
- **C:** wspólne uczenie na danych rzeczywistych i jednakowej liczbie przykładów syntetycznych;
- **D:** wspólne uczenie na danych rzeczywistych i całym zbiorze syntetycznym;
- **E:** uczenie na danych syntetycznych o liczebności odpowiadającej wycinkowi rzeczywistemu.

Domyślna konfiguracja obejmuje budżety 1%, 3%, 10%, 30% i 100% oraz ziarna 42, 43 i 44. Przykłady referencyjne pozostają w części treningowej i są wyłączone z walidacji oraz testu. Budżet 100% oznacza pełną dostępną pulę treningową.

Wycinki rzeczywiste są zagnieżdżone w obrębie danego ziarna: do stałych referencji dodawane są coraz dłuższe początkowe fragmenty jednej permutacji pozostałej puli treningowej. Wycinki syntetyczne wariantów C i E powstają analogicznie z jednej permutacji zbioru syntetycznego. Zwiększenie budżetu zachowuje wszystkie przykłady mniejszego wycinka. Przy włączonym losowaniu klas zasada dotyczy puli wejściowej; późniejsze losowanie ze zwracaniem nie gwarantuje zagnieżdżenia faktycznie wylosowanych wystąpień. Reguła zagnieżdżania dotyczy uruchomień tej wersji kodu; nie zmienia zapisanych wyników wcześniejszych uruchomień.

Kontrola danych bez ładowania modelu:

```powershell
python -m wiadomosci.eksperymenty --data-dir dane/tekst
```

Trening modeli:

```powershell
python -m wiadomosci.eksperymenty --data-dir dane/tekst --train
```

Dodatkowe losowanie klas ze zwracaniem:

```powershell
python -m wiadomosci.eksperymenty --data-dir dane/tekst --balanced --train
```

Oba tryby korzystają z jednego modułu `wiadomosci/rdzen.py`. Argument `--balanced` zmienia losowanie wystąpień treningowych, a nie procedurę oceny. Audyt i trening wyznaczają etapy tą samą funkcją oraz porównują kolejność identyfikatorów próbek przed rozpoczęciem uczenia. Wariant A obejmuje domyślnie jedną epokę na całym zbiorze syntetycznym i jedną na wycinku rzeczywistym; pozostałe warianty po jednej epoce.

Wspólna ocena porównuje logity pierwszych tokenów etykiet HAM i SPAM, zgodnie z podstawową procedurą eksperymentu. Końcowa klasyfikacja i strata korzystają z rozmiaru partii ustalonego dla treningu, a klasyfikacja walidacyjna z partii o rozmiarze 8. Strata jest średnią strat partii; maskowane są tokeny o identyfikatorze dopełnienia, wspólnym z `<|im_end|>`. Jest to nowa wspólna ścieżka również dla losowania klas, dlatego nie należy utożsamiać jej wyników z wcześniejszymi wynikami odrębnego modułu równoważącego. Pamięć ukończonego etapu wstępnego jest oddzielona według trybu, konfiguracji i danych oraz ziarna; jej pochodzenie i identyfikatory próbek są sprawdzane przed użyciem.

Zakres uruchomień ograniczają argumenty `--variants`, `--seeds` i `--budgets`. Parametry modelu, współczynniki uczenia i liczby epok określa plik konfiguracji. Wyniki, historie walidacji i identyfikatory próbek są zapisywane w `wyniki/tekst/<identyfikator>`.

Ponowne uruchomienie pomija ukończone konfiguracje. Argument `--retry-incomplete` rozpoczyna przerwany przebieg od modelu początkowego. Ukończony etap uczenia wstępnego wariantu A może być wykorzystany ponownie. Ocena dotyczy modelu po ostatniej epoce.

## 6. Eksperymenty obrazowe

Plik `konfiguracje/obrazy.json` określa źródła danych, parametry uczenia i augmentacji. Wszystkie warianty korzystają z tego samego źródła syntetycznego. Kolejka zapisuje ścieżkę wspólnego manifestu; wybrane klasy muszą odpowiadać manifestowi. Augmentacje obejmują kadrowanie 80–100%, odbicie poziome z prawdopodobieństwem 0,5, obrót do 15 stopni i zmianę jasności, kontrastu oraz nasycenia 0,2, bez dodatkowego szumu i degradacji JPEG. Kolejka jest przygotowywana następującym poleceniem:

```powershell
python -m obrazy.przygotuj_kolejke --workspace dane/obrazy --write
```

Interfejs treningowy jest uruchamiany poleceniem:

```powershell
python -m obrazy.uruchom --workspace dane/obrazy
```

Rozpoczęcie uczenia następuje w aplikacji. Argumenty `--variants` i `--budgets` umożliwiają ograniczenie przygotowywanej kolejki. Archiwum modeli i metadanych jest zapisywane w katalogu `experiments_archive`, a zestawienia wyników w `results`, we wskazanym katalogu roboczym.

## 7. Analizy wyników

### Metryki klasyfikacji

Agregacja wyników tekstowych:

```powershell
python -m analizy.wyniki_tekst --input wyniki/tekst/IDENTYFIKATOR/results.json
```

`IDENTYFIKATOR` oznacza nazwę katalogu podaną przez skrypt eksperymentów. Miara F0.5 jest obliczana bezpośrednio z liczebności TP, FP i FN.

Ocena zapisanych modeli obrazowych:

```powershell
python -m analizy.metryki_obrazy --root dane/obrazy --execute
```

Skrypt wykorzystuje pliki `best_model.pth` i listy testowe w `metadata.json`. Wyniki obejmują dokładność, precyzję, czułość i F1 uśrednione po klasach oraz mAP. Argument `--run-id` ogranicza ocenę do jednego modelu. Uruchomienie bez `--execute` przeprowadza wyłącznie kontrolę plików.

### Ranking obrazów do ręcznej kontroli

Syntetyczne obrazy można uporządkować malejąco według straty zapisanego klasyfikatora. Katalog `--model-dir` zawiera `metadata.json` oraz `best_model.pth`, a `--synthetic` podkatalogi klas zgodnych z modelem. Polecenie bez `--execute` sprawdza obecność plików i nazwy klas bez ładowania modelu:

```powershell
python -m analizy.ranking_straty --model-dir dane/obrazy/experiments_archive/IDENTYFIKATOR --synthetic dane/obrazy/obrazy_syntetyczne --output wyniki/ranking_straty --batch-size 32
```

Dodanie `--execute` uruchamia ocenę i zapisuje `ranking.csv` oraz `nieodczytane.json`. Wysoka strata wskazuje próbki do przeglądu, lecz nie przesądza o błędzie etykiety. Skrypt nie usuwa ani nie zastępuje obrazów; decyzje o jakości wymagają ręcznej oceny.

### Zróżnicowanie danych

Analiza długości, powtórzeń oraz koncentracji początków wiadomości i słownictwa:

```powershell
python -m analizy.sms --real dane/tekst/wiadomosci_rzeczywiste.csv --synthetic dane/tekst/wiadomosci_syntetyczne.csv --plots
```

Wyznaczenie wektorów cech obrazów i podobieństwa kosinusowego:

```powershell
python -m analizy.embeddingi --real dane/obrazy/obrazy_rzeczywiste --synthetic dane/obrazy/obrazy_syntetyczne --execute
python -m analizy.podobienstwo --input wyniki/embeddingi/embeddingi_resnet18.npz
```

Analiza podobieństwa wykorzystuje ResNet-18 z wagami ImageNet, bez dostrajania na danych sportowych. Porównania są wykonywane w obrębie klas, osobno wewnątrz zbiorów i pomiędzy źródłami. Wynik zbiorczy nadaje każdej klasie jednakową wagę.

## 8. Kontrola kodu

```powershell
python sprawdz_repo.py
python -m unittest discover -s tests -v
```

Kontrole obejmują składnię kodu i konfiguracji, rozłączność podziałów, skład wariantów, losowanie klas, agregację wyników oraz obliczenia podobieństwa. Testy są wykonywane bez pobierania modeli i uruchamiania treningów. Identyfikatory próbek, konfiguracje i wyniki pozostają zapisywane oddzielnie dla poszczególnych uruchomień.

Liczebności wygenerowanych obrazów odpowiadają liczebnościom każdej wybranej klasy rzeczywistej; dla domyślnego wykazu celem jest 4683 obrazów. Podzbiory treningowe wyznaczane są dynamicznie według poziomu budżetu. Nowe reguły podziału i wspólna ewaluacja SMS obowiązują w tej wersji kodu; nie zmieniają pochodzenia wyników zapisanych starszymi wersjami.
