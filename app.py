import streamlit as st
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import cv2
import os


uploaded_file = st.file_uploader('Choose a image file')



class resnet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 9)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x


model = resnet18()

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = pil2cv(image)

    # image = cv2.imread(uploaded_file)

    if image is None:
        print("Not open file:")

    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

    face_list = cascade.detectMultiScale(
        image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

    if len(face_list) > 0:
        for n, rectangle in enumerate(face_list):
            x, y, width, height = rectangle
            image = image[y: y + height, x: x + width]

            if image.shape[0] < 64:
                print("None")
            try:
                image = cv2.resize(image, (224, 224))

                image = cv2pil(image)

                img_array = np.array(image)
                # print("test2")
                # st.image(
                #     image, caption='upload images',
                #     use_column_width=True
                # )
                transform = transforms.Compose({
                    transforms.ToTensor()
                })
                image = transform(image)
                # print("test3")
                image= image.unsqueeze(dim=0)
                # print("test4")

                result = model(image)
                result = result.softmax(dim=1)
                # st.write(result)
                ans = torch.argmax(result)
                if ans == 0:
                    name = "AYAKA"
                elif ans == 1:
                    name = "MAKO"
                elif ans == 2:
                    name = "MAYA"
                elif ans == 3:
                    name = "MAYUKA"
                elif ans == 4:
                    name = "MIIHI"
                elif ans == 5:
                    name = "NINA"
                elif ans == 6:
                    name = "RIKU"
                elif ans == 7:
                    name = "RIMA"
                elif ans == 8:
                    name = "RIO"

                st.write(name)
            except:
                print(face_list)
                # print("test")

    else:
        print("not found face..")

    # image = Image.open(uploaded_file)
    # img_array = np.array(image)
    # st.image(
    #     image, caption='upload images',
    #     use_column_width=True
    # )

