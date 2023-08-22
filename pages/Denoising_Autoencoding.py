import streamlit as st
import torch
import requests
from PIL import Image
from io import BytesIO
from model.AutoEncoderClass import process_image, ImprovedAutoencoderNoPooling

st.set_page_config(
    page_title='Очищение документов от шумов с помощью автоэнкодера',  # Setting page title
    page_icon="📖"     # Setting page icon
)
st.write('## Очищение документов от шумов с помощью автоэнкодера')
device = 'cpu'
model = torch.load('weights/AutoEncoderWeights.pth', map_location=device)
model.eval()
source_img = st.file_uploader("##### Выберите файл...")

# Input box for user to enter image URL
image_url = st.text_input("##### Или введите URL изображения", "")

# Check if a URL is entered
if image_url:
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        source_img = BytesIO(response.content)
        
    except Exception as ex:
        st.error('Что-то не так со ссылкой')
    

col1, col2 = st.columns(2)

if source_img:
    input_image, output_image = process_image(source_img, model)
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
            st.image(output_image, caption='Фото без шума', use_column_width=True)  
        except Exception as ex:
            st.error(
                "Невозможно провести денойзинг.")
            st.error(ex)


