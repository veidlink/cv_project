import streamlit as st
import torch
from model.AutoEncoderClass import process_image, ImprovedAutoencoderNoPooling

st.set_page_config(
    page_title='Очищение документов от шумов с помощью автоэнкодера',  # Setting page title
    page_icon="🤖"     # Setting page icon
)
st.write('## Очищение документов от шумов с помощью автоэнкодера')
device = 'cpu'
model = torch.load('weights/AutoEncoderWeights.pth', map_location=device)
model.eval()
uploaded_file = st.file_uploader("##### Выберите файл...")
col1, col2 = st.columns(2)

if uploaded_file:
    input_image, output_image = process_image(uploaded_file, model)
    with col1:
        try:
            st.image(uploaded_file,
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


