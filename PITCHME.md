#### Inference in neural networks using low-precision arithmetic
<br>
<span style="color:gray">
  Krystyna Gajczyk<br>
  Jakub Pierewoj<br>
  Przemysław Przybyszewski<br>
  Adam Starak<br>
</span>

---

### Tensorflow
1. Otwarto-źródłowa platforma do obliczeń numerycznych rozwijana przez Google
2. Pozwala na wykonywanie obliczeń na procesorach oraz kartach graficznych
3. Umożliwia łatwą implementację sieci neuronowych
4. Posiada dobre wsparcie społeczności
<img src="https://lh3.googleusercontent.com/hIViPosdbSGUpLmPnP2WqL9EmvoVOXW7dy6nztmY5NZ9_u5lumMz4sQjjsBZ2QxjyZZCIPgucD2rhdL5uR7K0vLi09CEJYY=s688" width="200">

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
