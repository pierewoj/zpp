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

### ILSVRC ImageNet - Coroczna olimpiada dla komputerów
* Rozpoznawanie obiektów
* Lokalizowanie obiektów
* Zadania dotyczą zarówno obrazów tudzież filmów
* Co roku zbiór danych posiada około 1mln obrazów i 1000 klas

---

### ILSVRC ImageNet - Wyniki na przełomie ostatnich lat
<img src=http://students.mimuw.edu.pl/~as361021/wykres1.png>

---

### TO JEST MIEJSCE NA OBRAZEK NEURONU 

---
### GENEZA PROJEKTU
<img src="http://students.mimuw.edu.pl/~kg332118/Wybor-drogi.jpg">
---

### GENEZA PROJEKTU
<img src="https://www.motorola.ca/sites/default/files/library/storage/compare/images/product-moto-g-5.jpg">
<img src="http://f01.esfr.pl/foto/9/2620280922/ed3d312e9ae248c744c934193bfe45b2/siemens-kg49nai22,2620280922_7.jpg">
<img src="http://www.bowi.com.pl/pictures/8705_0.jpg">

---

### GENEZA PROJEKTU
### <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Intel-logo.svg/2000px-Intel-logo.svg.png">

---

### GENEZA PROJEKTU
### <img src="http://students.mimuw.edu.pl/~pp332493/xnornet.png">

---


### SIECI KONWOLUCYJNE
* Trzy typy warstw: konwolucyjna, 'pooling' oraz 'fully-connected'
* Kazdy neuron w warstwie konwolucyjnej jest polaczony z pewnym lokalnym podzbiorem danych wejsciowych.
### <img src="http://students.mimuw.edu.pl/~pp332493/CNN_layer.png">

---

### BINARNA KONWOLUCJA
### <img src="http://students.mimuw.edu.pl/~pp332493/binconv.png">

---

### CELE PROJEKTU

Główne:
* Weryfikacja, czy zwiekszanie ilości feature map poprawia dokladność zbinaryzowanej sieci neuronowej.
* Sprawdzenie, o ile średnio należy zwiekszyć tę ilość, aby osiagnąć taką samą jakość zbinaryzowanej sieci neuronowej jak oryginalnej.
---

### CELE PROJEKTU
Poboczne:
* Stworzenie operacji binarnej konwolucji jako operacji w Tensorflow
* Odtworzenie wynikow pracy opisanych przez tworcow XNORNET-a dla ImageNeta i AlexNeta.
* Sprawdzenie, czy podobne działania można powtórzyć na sieciach o innych strukturach

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
* Zgłoszenie prośby o dodanie naszej implementacji do TF

---
### TO JEST MIEJSCE NA WYPUNKTOWANIE UŻYTYCH ARCHITEKTUR

---

### AlexNet na oxford-102
<img src=http://students.mimuw.edu.pl/~as361021/AlexNet.png>

---

### Wyniki eksperymentów

<table>
  <tr>
    <th><th>
    <th>LeNet MNIST</th>
    <th>AlexNet oxford-102</th> 
    <th>ResNet-18 CIFAR-10</th>
  </tr>
  <tr>
    <th>Basic<th>
    <th>0.98</th>
    <th>0.85</th> 
    <th>0.84</th>
  </tr>
  <tr>
    <th>Binary weights vector<th>
    <th>0.98</th>
    <th>0.71</th> 
    <th>0.83</th>
  </tr>
  <tr>
    <th>Binary weights scalar<th>
    <th>0.98</th>
    <th>0.63</th> 
    <th>0.79</th>
  </tr>
</table>

---
### TU JEST MIEJSCE NA OPISANIE WYNIKÓW DLA RÓŻNYCH RATE 
