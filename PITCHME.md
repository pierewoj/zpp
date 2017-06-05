#### Inference in neural networks using low-precision arithmetic
<br>
<span style="color:gray">
  Krystyna Gajczyk<br>
  Jakub Pierewoj<br>
  Przemysław Przybyszewski<br>
  Adam Starak<br>
</span>
---
### Sieci neuronowe - co portafią

<img src=http://students.mimuw.edu.pl/~kg332118/Engel1.jpg>
---
### Sieci neuronowe - co portafią

<img src=http://students.mimuw.edu.pl/~kg332118/Engel-step1.jpg>
---
### Sieci neuronowe - co portafią

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
