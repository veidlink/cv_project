# cv_project

# Computer Vision Project
Elbrus Bootcamp | Phase-2 | Team Project

## 🦸‍♂️ Team
1. [Salman Chakaev](https://github.com/veidlink)
2. [Vladislav Filippov](https://github.com/Vlad1slawoo)

## 🎯 Task
Create a service for object detection using YOLOv5 and image denoising using a custom AutoEncoder class.

## 🪜 Contents

1. Pizza ingredient detection using YOLOv5
2. Brain tumor detection from photographs using YOLOv5
3. Document denoising using an autoencoder

## 🌐 Deployment
The service is implemented on [Streamlit](https://tumorencofood.streamlit.app/Pizza_Ingridients)

## 📚 Libraries 

```python
import torch
import PIL
import requests
import torch.nn as nn

from PIL import Image
from io import BytesIO
from torchvision import transforms
```

## 📚 Guide 
### How to run locally?

1. To create a Python virtual environment for running the code, enter:

    ``python3 -m venv my-env``.

2. Activate the new environment:

    * Windows: ```my-env\Scripts\activate.bat```
    * macOS and Linux: ```source my-env/bin/activate```

3. Install all dependencies from the *requirements.txt* file:

    ``pip install -r requirements.txt``..

##
UPD. In the future, galaxy segmentation from telescope photos using the Mask R-CNN model will be added.
