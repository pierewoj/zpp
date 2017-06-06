#### Inference in neural networks using low-precision arithmetic
<br>
<span style="color:gray">
  Krystyna Gajczyk<br>
  Jakub Pierewoj<br>
  Przemysław Przybyszewski<br>
  Adam Starak<br>
</span>
---
<img src=http://students.mimuw.edu.pl/~kg332118/plakat-gotowy.jpg>
---
### Sieci neuronowe - co potrafią

<img src=http://students.mimuw.edu.pl/~kg332118/Engel1.jpg>
---
### Sieci neuronowe - co potrafią

<img src=http://students.mimuw.edu.pl/~kg332118/Engel-step1.jpg>
---
### Sieci neuronowe - co potrafią

<img src=http://students.mimuw.edu.pl/~kg332118/Engel-step2.jpg>
---

### SIECI KONWOLUCYJNE
* Trzy typy warstw: konwolucyjna, 'pooling' oraz 'fully-connected'
* Kazdy neuron w warstwie konwolucyjnej jest polaczony z pewnym lokalnym podzbiorem danych wejsciowych.
### <img src="http://students.mimuw.edu.pl/~pp332493/CNN_layer.png">

---

### GENEZA PROJEKTU
### <img src="http://students.mimuw.edu.pl/~pp332493/xnornet.png">

---

### BINARNA KONWOLUCJA
### <img src="http://students.mimuw.edu.pl/~pp332493/binconv.png">

---

### CELE PROJEKTU

Glowne:
* Weryfikacja, czy zwiekszanie ilosci feature map poprawia dokladnosc zbinaryzowanej sieci neuronowej.
* Sprawdzenie, o ile srednio nalezy zwiekszyc te ilosc, aby osiagnac taka sama jakosc zbinaryzowanej sieci neuronowej jak oryginalnej.
Poboczne:
* Stworzenie operacji binarnej konwolucji jako operacje w Tensorflow (jedna ze zbinaryzowanymi wagami filtrow, druga ze zbinaryzowanymi wagami filtrow oraz danymi wejsciowymi).
* Odtworzenie wynikow pracy opisanych przez tworcow XNORNET-a dla ImageNeta i AlexNeta.
* Sprawdzenie, czy podobne dzialania mozna powtorzyc na sieciach o innych strukturach

---


---

### <img src="https://wiki.tum.de/download/attachments/25009442/tensor-flow_opengraph_h.png?version=1&modificationDate=1485888308193&api=v2" width="300">
* Otwarto-źródłowa platforma do obliczeń numerycznych rozwijana przez Google
* Pozwala na wykonywanie obliczeń na procesorach oraz kartach graficznych
* Umożliwia łatwą implementację sieci neuronowych
* Posiada dobre wsparcie społeczności


---

### Implementacja w Pythonie
* Utworzona poprzez złożenie istniejących operacji z TensorFlow
* Umożliwia uruchamianie na kartach graficznych oraz procesorach
* Używa operacji na liczbach zmiennoprzecinkowych
* Wydajność uczenia oraz inferencji zbliżona do standardowej konwolucji
* Używana przez nas do testów jakości klasyfikacji

---

### Implementacja w C++
* Utworzona nowa operacja w tensorflow
* Dostępna z API Pythonowego tak jak inne operacje
* Pozwala na optymalizacje, np użycie XNOR bitcount zamiast mnożenia i dodawania liczb zmiennoprzecinkowych
* Nie ma typów bitowych w TF?? tf.boolean
* Zgłoszenie prośby o dodanie naszej implementacji do TF
* jakieś przejście do eksperymentów
