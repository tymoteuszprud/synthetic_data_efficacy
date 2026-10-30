# Wyniki eksperymentów

Zestawienia liczbowe eksperymentów klasyfikacyjnych przedstawionych w pracy magisterskiej „Zbadanie efektywności syntetycznych zbiorów danych w uczeniu sieci neuronowych”.

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
- W eksperymencie obrazowym precyzja, czułość i F1 są uśrednione po klasach z jednakowymi wagami. `map_proc` oznacza średnią AP po klasach.
- W eksperymencie tekstowym klasą pozytywną jest SPAM. `f05_proc` oznacza miarę F0,5, a `tp`, `fp`, `tn`, `fn` oznaczają liczby rozpoznań w macierzy pomyłek względem tej klasy. Liczby te nie są procentami.
- W plikach podsumowujących każda średnia i każde odchylenie standardowe dotyczą trzech uruchomień. Odchylenie standardowe obliczono dla próby, z mianownikiem n-1. Agregacja następuje po obliczeniu metryk dla poszczególnych uruchomień; macierze pomyłek nie są łączone przed agregacją.
- Pliki mają kodowanie UTF-8 z BOM, separator przecinkowy i kropkę jako separator dziesiętny. Zachowano precyzję obliczeń; wartości w pracy są zaokrąglone.

## Zakres i pochodzenie

Wyniki obrazowe odpowiadają końcowemu zestawieniu wykorzystanemu w rozdziale 5 i dodatku C. Dla C przy budżecie 100% uwzględniono końcowe uruchomienia z jednakową liczbą rzeczywistych i syntetycznych przykładów treningowych. Wyniki tekstowe odpowiadają rozdziałowi 6; dodatkowe losowanie klas obejmuje przedstawiony w pracy wariant B.

Źródłami zestawienia obrazowego są zapisane metryki poszczególnych uruchomień (`metryki_uruchomienia.csv`) i ich końcowa agregacja (`metryki_finalne.csv`). Źródłem zestawień tekstowych są zapisane wyniki porównania uruchomień bez losowania i z losowaniem klas (`porownanie_75_uruchomien.csv`), zweryfikowane względem agregacji (`porownanie_25_konfiguracji.csv`). Metryki tekstowe sprawdzono także na podstawie liczb TP, FP, TN i FN. Pliki nie zawierają wag modeli ani zbiorów danych.
