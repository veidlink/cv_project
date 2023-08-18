import torch
import PIL
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
from torchvision import transforms as T
from PIL import Image
from io import BytesIO

# Model

# model_path = 'weights/FoodIngredientsWeights.pt'

st.set_page_config(
    page_title='Детекция опухулей мозга по фотографии с помощью YOLOv5',  # Setting page title
    page_icon="🤖"     # Setting page icon
)

st.write('## Детекция опухулей мозга по фотографии с помощью YOLOv5')

model = torch.hub.load(
'yolo', # пути будем указывать гдето в локальном пространстве
'custom', # непредобученная
path='weights/OpuholWeights.pt', # путь к нашим весам
source='local' # откуда берем модель – наша локальная
)

model.conf = st.select_slider('##### Установите порог доверия модели:',
                              options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


# # Function to download and display the image
# def save_image_from_link(url):
#     try:
#         response = requests.get(url)
#         source_img = Image.open(BytesIO(response.content))
#     except Exception as e:
#         st.error(f"Error: {e}")
#     return source_img


source_img = st.file_uploader(
        "##### Выберите файл")

# # User input for image URL
# url = st.text_input("Или предоставьте ссылку на него")

# source_img = save_image_from_link(url)

col1, col2 = st.columns(2)

if source_img:
        
    uploaded_image = PIL.Image.open(source_img)

    with col1:
    

        try:

            st.image(source_img,
                    caption="Загруженное фото",
                    )

            # model.load_state_dict(torch.load('/home/veidlink/ds_bootcamp/CVisionProject/web_page/weights/FoodIngredientsWeights.pt')['model'].state_dict())
        except Exception as ex:
            st.error(
                f"Загрузите фото.")

    with col2:

        try:

            result = model(uploaded_image)
            
            result = result.render()
            
            st.image(result,
                    caption="Фото с детекцией",
                    )

        except Exception as ex:
            st.error(
                "Невозможно провести детекцию.")
            st.error(ex)




