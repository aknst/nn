# Нейронная сеть, распознающая примитивные образы

Однослойная нейронная сеть, обученная по выборке MNIST.
Отчёт по работе доступен по [ссылке](https://disk.yandex.ru/i/VIzsddtkwoL7Hg).



## Демонстрация работы

![Alt Text](https://raw.githubusercontent.com/aknst/nn/main/digits.gif)


## Установка

1. [Скачать портативную версию программы](https://disk.yandex.ru/d/07oN9YcxSQPN1w)
2. Разархивировать, запустить `start.exe`

## Сборка

1. Скомпилировать `main.cpp`.
```
g++ -std=c++11 -O2 main.cpp -o nn.exe
```
2. Убедиться, что рядом с исполняемым файлом лежит обучающая выборка ([СКАЧАТЬ](https://disk.yandex.ru/d/3O5zLQ-zamp5fw)) `mnist_train.csv`.
3. Запустить `./nn.exe`, ввести гиперпараметры обучения. Результат — текстовый файл `model.txt` — обученная модель нейронной сети.
4. Открыть `app/start.pro` через среду Qt, собрать проект и запустить.
5. Загрузить `model.txt` в программу через ее меню.
