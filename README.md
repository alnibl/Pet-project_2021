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

Автомобили были получены путем сегментации, при помощи предобученной нейронной сети для сегментации изображений deeplabv3_resnet101 (Pytorch).

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

### При создании НС были использованы такие слои как Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate,Conv2DTranspose, Dropout; функции активации - LeakyReLU, relu, sigmoid, thanh; функции ошибки - binary_crossentropy, mse.


### Генератор.
За основу взята сеть U-Net, но с изменениями и дополениями.

Вход (None, 256, 384, 3) - изображение с машиной.

Выход (None, 256, 384, 3) - изображение без машины. Активационная функция - tanh (-1…1).

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

Выход  (None, 256, 384, 1) – карта признаков. Активационная функция - sigmoid (0…1).

### VGG19.
Предобученная сеть для получения карт признаков изображений. VGG19 не обучалась, так как веса я замарозил (trainable = False), но участвовала в подсчете loss всей модели.

Вход (None, 256, 384, 3) – изображение от генератора или чистый фон без машины.

Выход (None, 8, 12, 512) – карты признаков.

### Функция объединяющая генератор, дискриминатор и VGG19:

![](/images/gan.png)

### Общая ошибка модели строится из:
1) ошибки дискриминатора, который оценивает качество сгенерированного генератором изображения

2) ошибки предобученной сети VGG19 на полученных фичах от Y_train и X_pred

3) ошибки генератора между Y_train и X_pred

### Общая функция работы нейронной сети:

    batch_size =8

    def train(generator, discriminator, gan, vgg, epochs, batch_size):

      curTime = time.time() # засекаю время  

      # загружаю листы с ошибками
      g_loss_list0 = list(np.load('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list0.npy')) # Массив значений ошибки генератора
      g_loss_list1 = list(np.load('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list1.npy')) # Массив значений ошибки генератора
      g_loss_list2 = list(np.load('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list2.npy')) # Массив значений ошибки генератора
      g_loss_list3 = list(np.load('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list3.npy')) # Массив значений ошибки генератора
      d_loss_list = list(np.load('/content/drive/MyDrive/Модели/Ход обучения/d_loss_list.npy'))   # Массив значений ошибки дескриминатора
      #d_acc_list = list(np.load('/content/drive/MyDrive/Модели/Ход обучения/d_acc_list.npy'))    # Массив значений точности дескриминатора

      for epoch in range(epochs): 

        with tqdm(total=ln) as pbar: #для отслеживания создаем progressbar
          for batch in range(ln//batch_size):

            #для тренировки дискриминатора применяем label smoothing (для борьбы с переобучением)
            #размерность Y: (размер батча, высота последнего слоя дискриминатора, ширина, 1)
            y_real = np.ones((batch_size, *discriminator.output_shape[1:])) - np.random.random_sample((batch_size, *discriminator.output_shape[1:]))*0.2 # значения от 0.8 до 1
            y_fake = np.random.random_sample((batch_size, *discriminator.output_shape[1:]))*0.2 # значения от 0 до 0.2

            x_tr, y_tr = image_paste_x_train() # функция формирования x_train и y_train по батчам. Получаю numpy изображения размером batch_size.

            cars = (x_tr/127.5)-1               # изображения с машиной нормализую
            image_reference = (y_tr/127.5)-1    # изображения без машины нормализую

            fake_imgs = generator.predict(cars) # удаляем машину генератором

            #тренируем дискриминатор       
            d_loss_real = discriminator.train_on_batch([image_reference, image_reference], y_real) # дискрминатор обучаю на реальных изображениях (чистый фон)
            d_loss_fake = discriminator.train_on_batch([fake_imgs,image_reference], y_fake) # дискриминатор обучаю на фэйковых изображениях (изображение после генератра)
            d_loss_total = 0.5*np.add(d_loss_real, d_loss_fake) # ошибка дискриминатора

            #тренируем генератор
            real_features = vgg.predict(image_reference) # с помощью vgg19 получаю фичи от изображенияй чистого фона 
            y_real = np.ones((batch_size, *discriminator.output_shape[1:])) # y_true для дискриминатора в общей модели gan

            g_loss = gan.train_on_batch([cars, image_reference], [y_real, image_reference, real_features]) # обучаю gan

            g_loss_list0.append(g_loss[0]) # Добавляем в массив значений ошибок генератора g_loss
            g_loss_list1.append(g_loss[1]) # Добавляем в массив значений ошибок генератора g_loss
            g_loss_list2.append(g_loss[2]) # Добавляем в массив значений ошибок генератора g_loss
            g_loss_list3.append(g_loss[3]) # Добавляем в массив значений ошибок генератора g_loss
            d_loss_list.append(d_loss_total[0]) # Добавляем в массив значений ошибок дискриминатора
            #d_acc_list.append(d_loss_total[1]) # Добавляем в массив значений точности дискриминатора
            pbar.update(batch_size) #обновляем progressbar
            pbar.set_description("Epoch: {}/{}, Acc: {} %, Dis loss: {}, Gen loss:{}".format(epoch+1, epochs, 100*d_loss_total[1], d_loss_total[0], g_loss))

        # каждые 3 эпохи вывожу результат
        if ((epoch % 3 == 0) | (epoch == epochs-1)):
          imege_pred(generator, epoch, 2, x_tr, y_tr) # отображаю результат, 2 строки

          plt.figure(figsize=(20,20))
          plt.subplot(3, 2, 1)
          plt.plot(g_loss_list0, label="Ошибка 0")
          plt.legend()
          plt.subplot(3, 2, 2)
          plt.plot(g_loss_list1, label="Ошибка 1")
          plt.legend()
          plt.subplot(3, 2, 3)
          plt.plot(g_loss_list2, label="Ошибка 2")
          plt.legend()
          plt.subplot(3, 2, 4)
          plt.plot(g_loss_list3, label="Ошибка 3")
          plt.legend()
          plt.subplot(3, 2, 5)
          plt.plot(d_loss_list, label="Ошибка дискриминатора")
          plt.legend()
          #plt.subplot(3, 2, 6)
          #plt.plot(d_acc_list, label="Точность дискриминатора")
          #plt.legend()
          plt.savefig('/content/drive/MyDrive/Модели/Ход обучения/Графики ошибок_lr_1e-5_LW_1_200_100.png', dpi = 100) # сохраняю графики
          plt.show()
          plt.close() # Завершаем работу с plt


        # Сохраняю модель генератора, дискриминатора и весь gan
        gen.save('/content/drive/MyDrive/Модели/Gen_7_256_384.h5')
        dis.save('/content/drive/MyDrive/Модели/Dis_7_256_384.h5')
        #gan.save('/content/drive/MyDrive/Модели/GAN_7_256_384.h5')

        # Сохраняю историю ошибок
        np.save('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list0', np.array(g_loss_list0))
        np.save('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list1', np.array(g_loss_list1))
        np.save('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list2', np.array(g_loss_list2))
        np.save('/content/drive/MyDrive/Модели/Ход обучения/g_loss_list3', np.array(g_loss_list3))
        np.save('/content/drive/MyDrive/Модели/Ход обучения/d_loss_list', np.array(d_loss_list))
        #np.save('/content/drive/MyDrive/Модели/Ход обучения/d_acc_list', np.array(d_acc_list))

        print('GAN осохранена на {} эпохе. Время от старта: {} мин. {} сек.'.format(epoch, round(((time.time() - curTime)// 60)), round((time.time() - curTime) % 60)))
      print('Время работы: {} мин. {} сек.'.format(round(((time.time() - curTime)// 60)), round((time.time() - curTime) % 60)))

### Optimizer - Adam. Loss - binary_crossentropy и mse. Параметры loss_weights, learning_rate менял в процессе обучения.
    gen = build_generator() #создаем генератор

    dis = build_discriminator() #создаем дискриминатор
    dis.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4, beta_1=0.5), metrics=['accuracy'])

    vgg = build_vgg() #создаем vgg модель

    gan = build_gan(dis, gen, vgg) #создаем gan
    gan.compile(loss=['binary_crossentropy', 'mse', 'mse'], loss_weights=[1,200,100], optimizer=Adam(learning_rate=1e-4, beta_1=0.5)) #вклад разницы между пикселями в 200 раз         больше и в feature loss в 100 раз больше, чем вклад от GAN.

### Проблемму нехватки ОЗУ в colab я решил.
Написал функцию image_paste_x_train(), которая формирует X_train прямо во время обучения нейронной сети.

Что происходит в функции image_paste_x_train():

ImageDataGenerator берет картинки авто и фона из google colab (или google drive), происходит аугментация по заданным параметрам,
дальше batch_size картинок преобразовываются из numpy в формат изображений RGBA, чтобы дальше производить наложение авто на фон. 
Затем ещё раз меняется размер авто, далее с помощью функции past машина налаживается на фон, координаты наложения авто выбираются случайно (random). Потом, на итоговое получившееся изображение машины на фоне, применяются еще дополнительные преобразования из библиотеки Аlbumentations [(https://albumentations.ai/)](https://albumentations.ai/). Благодаря этой библиотеке на финальное фото рандомно налаживаются эффекты дождя, солнца, тумана, тени, изменяются значение RGB картинки, применяется размытие к изображениям, также RandomCrop и Resize. Затем данные преобразовываются в numpy и передаются в функцию обучения сети – train. Сеть обучается 'побатчево' в цикле, методом train_on_batch.

### Графики ошибок.
![](/images/loss.png)

### Пример работы.
![](/images/pred_1.png)
![](/images/pred_3.png)
![](/images/pred_55.jpg)![](/images/pred_66.jpg)

### [Пример работы нейронной сети на ВИДЕО тут.](https://youtu.be/BDtDp4iygtQ)

### Какие выводы после всей работы и что можно сделать лучше?

- Для обучения нейронной сети использовать максимально большую базу фонов: 10 – 20 - 50 тыс. изображений и более. Все упирается в длительность обучения и имеющиеся ресурсы “железа”. 

С имеющейся у меня НС, ресурсами colab pro (видеокарта Tesla T4), скоростью обучения, если бы база фонов была 25 тыс. изображений, разрешение картинки 256 на 384, то обучение на 500 эпохах заняло бы около 50 дней.

- Если стоит задача удалить авто из тематического видео, например, авто на фоне гор, то и базу фонов нужно формировать в основном из фонов гор. 
Фоны должны соответствовать тематике видео, из которого планируется удалять авто.

- Авто для наложения на фон должны быть сфотографированы из максимально возможного числа ракурсов.


### Что я бы ещё попробовал?

- Генератор:

Увеличил количество слоев/карт признаков в генераторе.

Увеличил количество веток сжатия и разжатия. Каждая ветка была бы со своим уникальным ядром свертки (kernel_size=1, kernel_size=2, … kernel_size=7).

Убрал бы подсчет ошибки по выходу из генератора. Генератор бы отдавал изображение дискриминатору, и ошибка считалось бы только по дискриминатору (binary_crossentropy) и по VGG (mse). Тут стоило бы поэкспериментировать с архитектурой дискриминатора.

- Максимально увеличил базу.

- Использовал Albumentations во всю мощь (больше трансформаций).

- Арендовать мощную видеокарту и обучить НС.


### Общее время обучения нейронной сети - 4,6 дня, видеокарта - Tesla T4.

### При работе над кодом использовались такие библиотеки как:
### Keras, Tensorflow, PyTorch, PIL, Numpy, Albumentations, CV2, Matplotlib.

### [Весь код нейронной сети можно помотреть тут.](https://github.com/alnibl/Portfolio/blob/main/Копия_блокнота_"GAN_7_ipynb".ipynb)

