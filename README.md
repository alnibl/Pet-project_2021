# Проект_1
# "Удаление автомобиля из видео"

## 1. Тема, описание задачи.
В дипломной работе стоит задача удалить автомобиль из видео при помощи нейронной сети, заменив авто ландшафтом (окружающей средой). 
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
Фото парсились с сайта http://www.motorpage.ru.
![](/images/16.jpg)

Автомобили были получены путем сегментации, при помощи предобученной нейронной сети для сегментации изображений deeplabv3_resnet101 (библиотека Pytorch).

Авто попиксельно выделялись и отдельно сохранялись в формате PNG, для дальнейшего наложения их на изображения фонов (для получения обучающей выборки).
![](/images/45.png)

## 3. Параметризация данных.

![](/images/seg_1.png)

![](/images/seg_2.png)

Если очень упрощенно описать, то:

X_train для НС это будут фото авто на фоне, а Y_train, это фото фона без автомобилей. 

Нейронная сеть, получая на вход фото авто на фоне, в ходе обучения, в результате back propagation, будет стремиться сделать loss минимальной, а значит изображение на выходе из генератора после обучения сети должно быть максимально похоже на y_train – то есть на картинку фона без автомобиля.
Следовательно, x_train отличается от y_train только тем, что в x_train есть авто, а в y_train его нету. Фон будет идентичным как в x_train, так и в y_train.

### [Пример кода тут.](https://github.com/alnibl/Portfolio/blob/main/"Эксперементы_с_GAN_lodki_ipynb"".ipynb)

## [4. Исследование нейронной сети.](https://github.com/alnibl/Portfolio/blob/main/Копия_блокнота_"GAN_7_ipynb".ipynb)
![](/images/4.png)

### При создании НС были использованы такие слои как Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate,Conv2DTranspose, Dropout: функции активации - LeakyReLU, relu, sigmoid, thanh; функции ошибки - binary_crossentropy, mse.


### Генератор.
За основу взята сеть U-Net, но с изменениями и дополениями.

Вход (None, 256, 384, 3) - изображение с машиной.

Выход (None, 256, 384, 3) - изображение без машины. Функция tanh (-1…1).

Функция генератора:

    def build_generator():
        filters  = 32 #минимальное число фильтров

        def en_conv2d(layer_input, filters, k_size_1=4, k_size_2=4, strides=2, bn=True, maxp=False): #слой с понижением разрешения.
            '''
            layer_input - слой на вход.
            filters- количество фильтров
            k_size_1 - размер ядра свертки
            k_size_2 - размер ядра свертки
            strides - какой strides применять в слое Conv2D
            bn - BatchNormalization
            maxp - MaxPooling2D
            '''
            en = Conv2D(filters, kernel_size=k_size_1, strides=1, padding='same')(layer_input) 
            en = LeakyReLU(alpha=0.2)(en)
            en = Conv2D(filters, kernel_size=k_size_2, strides=strides, padding='same')(en)
            en = LeakyReLU(alpha=0.2)(en)
            if bn:
                en = BatchNormalization(momentum=0.8)(en)
                if maxp:
                  en = MaxPooling2D(2)(en)
            return en

        def de_conv2d(layer_input, skip_input, filters, k_size_1=4, k_size_2=4, dropout_rate=0): #слой с повышением разрешения
            '''
            layer_input - слой на вход
            skip_input -  предыдущий слой от слоя с понижением разрешения (conv2d)
            filters - количество фильтров
            k_size_1 - размер ядра свертки
            k_size_2 - размер ядра свертки
            dropout_rate - применять ли dropout
            '''        
            de = UpSampling2D(size=2)(layer_input) #увеличивам разрешение в 2 раза
            de = Conv2D(filters, kernel_size=k_size_2, strides=1, padding='same', activation='relu')(de) #strides=1, padding='same',  поэтому разрешение сохраняется
            de = Conv2D(filters, kernel_size=k_size_1, strides=1, padding='same', activation='relu')(de) #strides=1, padding='same',  поэтому разрешение сохраняется
            if dropout_rate:
                de = Dropout(dropout_rate)(de)
            de = BatchNormalization(momentum=0.8)(de)
            de = Concatenate()([de, skip_input]) #соединяем skip-слой от conv2d (слой с понижением разрешения) и слой от deconv2d (слой с повышением разрешения)
            return de


        e0 = Input(shape=img_shape, name="condition") #входное изображение (условие)

        # Ветка 1, где понижается разрешение
        e1 = en_conv2d(e0, filters, bn=False)
        e2 = en_conv2d(e1, filters*2)
        e3 = en_conv2d(e2, filters*4) #чем меньше размер карт активаций
        e4 = en_conv2d(e3, filters*8) #тем больше должно быть фильтров в сверточном слое
        e5 = en_conv2d(e4, filters*8)
        e6 = en_conv2d(e5, filters*8)
        e7 = en_conv2d(e6, filters*8)

        # Ветка 2, где понижается разрешение
        ee1 = en_conv2d(e0, filters, k_size_1=2, k_size_2=2, bn=False)
        ee2 = en_conv2d(ee1, filters*2, k_size_1=2, k_size_2=2, strides=1, maxp=True)
        ee3 = en_conv2d(ee2, filters*4, k_size_1=2, k_size_2=2, strides=1, maxp=True) #чем меньше размер карт активаций
        ee4 = en_conv2d(ee3, filters*8, k_size_1=2, k_size_2=2, strides=1, maxp=True) #тем больше должно быть фильтров в сверточном слое
        ee5 = en_conv2d(ee4, filters*8, k_size_1=2, k_size_2=2, strides=1, maxp=True)
        ee6 = en_conv2d(ee5, filters*8, k_size_1=2, k_size_2=2, strides=1, maxp=True)
        ee7 = en_conv2d(ee6, filters*8, k_size_1=2, k_size_2=2, strides=1, maxp=True)

        e_cont = Concatenate()([e7, ee7]) #соединяю 2 слоя вместе (от ветки 1 и ветки 2)
        e_cont = Conv2D(filters*8, kernel_size = 4, strides=1, padding='same', activation='relu',  name='d_cont') (e_cont) # прохожусь ещё раз сверткой

        # Ветка 1 повышения разрешения
        d1 = de_conv2d(e_cont, e6, filters*8)
        d2 = de_conv2d(d1, e5, filters*8) 
        d3 = de_conv2d(d2, e4, filters*8) #чем больше размер карт активаций
        d4 = de_conv2d(d3, e3, filters*4) #тем меньше должно быть фильтров в сверточном слое
        d5 = de_conv2d(d4, e2, filters*2)
        d6 = de_conv2d(d5, e1, filters)

        # Ветка 2 повышения разрешения
        dd1 = de_conv2d(e_cont, ee6, filters*8, k_size_1=2, k_size_2=2)
        dd2 = de_conv2d(dd1, ee5, filters*8, k_size_1=2, k_size_2=2) 
        dd3 = de_conv2d(dd2, ee4, filters*8, k_size_1=2, k_size_2=2) #чем больше размер карт активаций
        dd4 = de_conv2d(dd3, ee3, filters*4, k_size_1=2, k_size_2=2) #тем меньше должно быть фильтров в сверточном слое
        dd5 = de_conv2d(dd4, ee2, filters*2, k_size_1=2, k_size_2=2)
        dd6 = de_conv2d(dd5, ee1, filters, k_size_1=2, k_size_2=2)

        d7 = UpSampling2D(size=2)(d6)
        output_img_1 = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh', name='output_img_1')(d7) # для ветки 1 - UpSampling2D и Conv2D
        output_img_2 = Conv2DTranspose(3, kernel_size=2, strides=2, padding='same', activation='tanh', name='output_img_2')(dd6) # для ветки 2 - Conv2DTranspose
        output_img_cont = Concatenate(name='output_img_cont')([output_img_1, output_img_2]) # соединяю вместе

        output = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh', name='Gen_output')(output_img_cont) #интенсивность должна быть от -1 до 1, поэтому используем tanh

        return Model(e0, output, name="Generator")

### Дискриминатор.
Урезанная версия генератора(U-Net).

Вход-1 (None, 256, 384, 3) - изображение от генератора (фейк) или реальное.

Вход-2 (None, 256, 384, 3) - изображение условие (фон без машины).

Выход  (None, 256, 384, 1) – карта признаков. Функция sigmoid (0…1).

### VGG19.
Предобученная сеть для получения карт признаков изображений. Не обучалась(заморозил веса), но участвовала в подсчете loss сети.

Вход (None, 256, 384, 3) – изображение от генератора или чистый фон без машины.

Выход (None, 8, 12, 512) – карты признаков.

### Функция объединяющая генератор, дискриминатор и VGG19:

![](/images/gan.png)

### Общая ошибка сети строится из:
1) ошибки дискриминатора, который оценивает качество сгенерированного генератором изображения

2) ошибки предобученной сети VGG19 на полученных фичах от Y_train и X_pred

3) ошибки генератора между Y_train и X_pred


### Проблемму нехватки ОЗУ в colab я решил.
Написал функцию image_paste_x_train(), которая формирует X_train прямо во время обучения нейронной сети.

Что происходит в функции image_paste_x_train():

ImageDataGenerator берет картинки авто и фона с colab (или google диска), происходит аугментация по заданным параметрам,
дальше batch_size картинок из генератора преобразовываются из numpy в формат изображений, чтобы дальше производить наложение авто на фон. 
Затем ещё раз меняется размер авто, далее с помощью функции past машина налаживается на фон, координаты наложения авто выбираются случайно (random), 
данные преобразовываются в numpy и передаются в функцию обучения сети – train, сеть обучается в цикле методом train_on_batch.

Для преобразования фона я применил библиотеку Аlbumentations [(https://albumentations.ai/)](https://albumentations.ai/) благодаря которой на фото налаживал эффекты дождя, солнца, тумана, тени, изменял значение RGB картинок, применял размытие к изображениям, применял RandomCrop и Resize.

### Графики ошибок.
![](/images/loss.png)

### Пример работы.
![](/images/pred_1.png)
![](/images/pred_3.png)
![](/images/pred_55.jpg)![](/images/pred_66.jpg)

### [Пример работы нейронной сети. Видео.](https://youtu.be/BDtDp4iygtQ)

### [Весь код нейронной сети можно помотреть тут.](https://github.com/alnibl/Portfolio/blob/main/Копия_блокнота_"GAN_7_ipynb".ipynb)

