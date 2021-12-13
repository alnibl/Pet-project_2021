# Проект_1 "Удаление автомобиля из видео"

## 1. Тема, описание задачи.
В дипломной работе стоит задача взять фрагмент видео, на котором изображен автомобиль, и удалить этот автомобиль из видео, заменив авто ландшафтом (окружающей средой). 
Первостепенно стоит задача убрать автомобиль из фото, затем из видео. 
Второстепенно (в идеале), хотелось бы заменить автомобиль качественным ландшафтом, чтобы исчезновение авто из видео стало как можно менее заметным.

## [2. База.](https://github.com/alnibl/Portfolio/blob/main/Сбор%2C_анализ_и_парсинг_данных_ipynb"".ipynb)
### Фон.
Общее количество фото фонов для обучения нейронной сети - 2528 изображений.

Из них:

- получены путем парсинга сайта https://oir.mobi – 1420 шт.
![](/images/130.jpg)
- взяты фреймы из видео YouTube – 406 шт.
![](/images/0_38.jpg)
- фото из баз в интернете(частично использованы базы DeepLoc, DeepLocCross, Freiburg Street Crossing, FRIDA ) - 702 шт.

 ![](/images/Image_79.png)

### Автомобили.
Общее количество фото - 2360 штук.
Фото парсились с сайта http://www.motorpage.ru
![](/images/16.jpg)

Автомобили были получены путем сегментации, при помощи предобученной нейронной сети для сегментации изображений deeplabv3_resnet101 (библиотека Pytorch).

Авто попиксельно выделялись и отдельно сохранялись в формате PNG, для дальнейшего наложения их на изображения фонов (для получения обучающей выборки).
![](/images/45.png)

## 3. Параметризация данных.
Если очень упрощенно описать, то:

X_train для НС это будут фото авто на фоне, а Y_train, это фото фона без автомобилей. 

Нейронная сеть, получая на вход фото авто на фоне, в ходе обучения, в результате back propagation, будет стремиться сделать loss минимальной, а значит изображение на выходе из генератора после обучения сети должно быть максимально похоже на y_train – то есть на картинку фона без автомобиля.
Следовательно, x_train отличается от y_train только тем, что в x_train есть авто, а в y_train его нету. Фон будет идентичным как в x_train, так и в y_train.

### Общая ошибка сети строится из:
1) ошибки дискриминатора, который оценивает качество сгенерированного генератором изображения

2) ошибки предобученной сети VGG19 на полученных фичах от Y_train и X_pred

3) ошибки генератора между Y_train и X_pred
