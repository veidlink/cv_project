import torch
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title='Детекция ингридиентов пиццы с помощью YOLOv5',  # Setting page title
    page_icon="🍕"     # Setting page icon
)

st.write('## Детекция ингридиентов пиццы с помощью YOLOv5')

model = torch.hub.load(
'yolo', # пути будем указывать гдето в локальном пространстве
'custom', # непредобученная
path='weights/FoodIngredientsWeights.pt', # путь к нашим весам
source='local' # откуда берем модель – наша локальная
)
model.conf = st.select_slider('Установите порог доверия модели:',
                              options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

source_img = st.file_uploader(
        "##### Выберите файл")

# Input box for user to enter image URL
image_url = st.text_input("##### Или введите URL изображения", "")

# Check if an image is uploaded
if source_img is not None:
    # Read the image from uploaded file
    source_img = source_img
    # Process and display the image

# Check if a URL is entered
elif image_url:
    # Download the image from the URL
    response = requests.get(image_url)
    source_img = Image.open(BytesIO(response.content))

col1, col2 = st.columns(2)
if source_img:
    with col1:
        try:
            st.image(source_img,
                    caption="Загруженное фото",
                    )
        except Exception as ex:
            st.error(
                f"Загрузите фото.")
    with col2:
        try:
            result = model(source_img)
            result = result.render()
            st.image(result,
                    caption="Фото с детекцией",
                    )
        except Exception as ex:
            st.error(
                "Невозможно провести детекцию.")
            st.error(ex)




