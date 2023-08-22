# cv_project

# Computer Vision Project
Elbrus Bootcamp | Phase-2 | Team project

## 🦸‍♂️Команда
1. [Салман Чакаев](https://github.com/veidlink) 
2. [Владислав Филиппов](https://github.com/Vlad1slawoo)

## 🎯 Задача
Создать сервис для задания детекция с помощью YOLOv5, а также очищения изображения от шума с помощью собственного класса AutoEncoder.

## 🪜 Содержание

1. Детекция ингридиентов пиццы с помощью YOLOv5
2. Детекция опухулей мозга по фотографии с помощью YOLOv5
3. Очищение документов от шумов с помощью автоэнкодера

## 🌐 Деплоймент
Сервис реализован на [Streamlit](https://tumorencofood.streamlit.app/Pizza_Ingridients)

## 📚 Библиотеки 

```typescript
import torch
import PIL
import requests
import torch.nn as nn

from PIL import Image
from io import BytesIO
from torchvision import transforms
```

## 📚 Гайд 
### Как запустить локально?

1. Чтобы создать виртуальную среду Python (virtualenv) для запуска кода, введите:

    ``python3 -m venv my-env``.

2. Активируйте новую среду:

    * Windows: ```my-env\Scripts\activate.bat```
    * macOS и Linux: ```source my-env/bin/activate```

3. Установите все зависимости из файла *requirements.txt*:

    ``pip install -r requirements.txt``..
