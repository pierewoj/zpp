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
    <th>???</th>
  </tr>
</table>
