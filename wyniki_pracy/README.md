# Wyniki eksperymentów

Zestawienia liczbowe eksperymentów klasyfikacyjnych przedstawionych w rozdziałach 5 i 6 oraz dodatku C pracy magisterskiej „Zbadanie efektywności syntetycznych zbiorów danych w uczeniu sieci neuronowych”.

## Zawartość

| Plik | Zakres | Liczba wierszy danych |
| --- | --- | --- |
| `obrazy_uruchomienia.csv` | Wyniki klasyfikacji obrazów dla poszczególnych uruchomień | 75 |
| `obrazy_podsumowanie.csv` | Średnie i odchylenia standardowe wyników obrazowych | 25 |
| `sms_uruchomienia.csv` | Wyniki podstawowego eksperymentu tekstowego | 75 |
| `sms_podsumowanie.csv` | Średnie i odchylenia standardowe podstawowych wyników tekstowych | 25 |
| `sms_losowanie_B_uruchomienia.csv` | Wyniki dodatkowego eksperymentu z losowaniem klas w wariancie B | 15 |
| `sms_losowanie_B_podsumowanie.csv` | Średnie i odchylenia standardowe wyników z losowaniem klas | 5 |

## Oznaczenia i format

- `wariant`: A - uczenie na danych syntetycznych, a następnie dostrajanie na rzeczywistych; B - uczenie na danych rzeczywistych; C - wspólne uczenie na wycinkach danych rzeczywistych i syntetycznych; D - wspólne uczenie na wycinku danych rzeczywistych i pełnym dostępnym zbiorze syntetycznym; E - uczenie na danych syntetycznych.
- `budzet_proc`: oznaczenie poziomu budżetu danych zgodne z definicjami w rozdziale 3. Poziom 100 oznacza pełną pulę treningową danych rzeczywistych po wydzieleniu walidacji i testu.
- `ziarno`: ziarno losowości, odpowiednio 42, 43 lub 44.
- Kolumny zakończone na `_proc` zawierają metryki w procentach, w skali od 0 do 100.
- W eksperymencie obrazowym precyzja, czułość i F1 są uśrednione po klasach z jednakowymi wagami. `map_proc` oznacza średnią precyzję uśrednioną po klasach (mAP).
- W eksperymencie tekstowym klasą pozytywną jest SPAM. `f05_proc` oznacza miarę F0,5. Kolumny `tp`, `fp`, `tn` i `fn` zawierają odpowiednio liczby poprawnych rozpoznań SPAM, błędnych rozpoznań SPAM, poprawnych rozpoznań HAM i błędnych rozpoznań HAM. Liczby te nie są procentami.
- W plikach podsumowujących każda średnia i każde odchylenie standardowe dotyczą trzech uruchomień. Odchylenie standardowe obliczono dla próby, z mianownikiem n-1. Agregacja następuje po obliczeniu metryk dla poszczególnych uruchomień; macierze pomyłek nie są łączone przed agregacją.
- Pliki mają kodowanie UTF-8 z BOM, separator przecinkowy i kropkę jako separator dziesiętny. Zachowano precyzję obliczeń; wartości w pracy są zaokrąglone.

