import streamlit as st
import os, json, cv2, random
import numpy as np

from PIL import Image
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os, json, cv2, random

st.set_page_config(
    page_title='Сегментация галактик силами модели семейства Detectron - MASK R-CNN',  # Setting page title
    page_icon="🌌"     # Setting page icon
)

st.write('## Сегментация галактик силами модели семейства Detectron - MASK R-CNN')


yaml_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(yaml_path))
# Устанавливаем порог для детекции: если уровень доверия меньше порога, детекция не состоится
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = st.select_slider('##### Установите порог доверия модели:',
                              options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# Загружаем модель
cfg.MODEL.WEIGHTS = 'weights/GalaxySegmentationWeights.pth'
cfg.MODEL.DEVICE = 'cpu'

# Создаем объект предиктора
predictor = DefaultPredictor(cfg)

source_img = st.file_uploader(
        "##### Выберите файл")

col1, col2 = st.columns(2)

if source_img:
    uploaded_image = Image.open(source_img)
    with col1:
        try:
            st.image(uploaded_image,
                    caption="Загруженное фото",
                    )
        except Exception as ex:
            st.error(
                f"Загрузите фото.")
    with col2:
        try:
            im = np.array(uploaded_image)
            outputs = predictor(im[:, :, ::-1])
            # Визуализация результата сегментации
            v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            st.image(out.get_image()[:, :, ::-1], caption='Результат сегментации', use_column_width=True)
        except Exception as ex:
            st.error(
                "Невозможно провести детекцию.")
            st.error(ex)

uploaded_image = Image.open(source_img)
im = np.array(uploaded_image)
outputs = predictor(im[:, :, ::-1])
outputs
