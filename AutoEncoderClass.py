import torch
import torch.nn as nn
import torch

from PIL import Image
from torchvision import transforms

class ImprovedAutoencoderNoPooling(nn.Module):
    def __init__(self):
        super(ImprovedAutoencoderNoPooling, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # stride=2 заменяет MaxPool2d
            nn.Tanh(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # stride=2 заменяет MaxPool2d
            nn.Tanh(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(64)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return x


def process_image(image_path, model):

    ''' 
    Код выполняет следующие действия: 
 
1. Загружает изображение с помощью функции  Image.open(image_path) . 
2. Применяет преобразования к изображению, включая изменение размера до 512x512 пикселей 
и преобразование в тензор с помощью  transforms.Compose()  и  transforms.ToTensor() . 
3. Преобразует тензорное изображение, используя модель  model , с помощью оператора  
with torch.no_grad(): . Здесь  torch.no_grad()  указывает, 
что градиенты не нужны для данной операции. 
4. Преобразует тензорный вывод обратно в изображение с помощью  transforms.ToPILImage() . 
5. Возвращает исходное изображение и выходное изображение. 
    
    '''
    def process_image(image_path, model):
    input_image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    tensor_image = transform(input_image).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(tensor_image)
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    return input_image, output_image
