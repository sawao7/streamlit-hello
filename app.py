import streamlit as st
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import cv2
import os

# モデルの読み込み
class resnet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18()
        self.fc = nn.Linear(1000, 9)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
model = resnet18()
param_load = torch.load("model.prm",map_location=torch.device('cpu'))
model.load_state_dict(param_load)
model = model.eval()


# 関数の定義
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

def opencv_resize_image(path):
    image = Image.open(path)
    image = pil2cv(image)
    if image is None:
        print("none")
        return None
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))
    if len(face_list) > 0:
        for n, rectangle in enumerate(face_list):
            x, y, width, height = rectangle
            image = image[y: y + height, x: x + width]

            if image.shape[0] < 64:
                print("None")
            try:
                image = cv2.resize(image, (224, 224))

                image = cv2pil(image)

                return image

            except:
                print("None")
                return None
    else:
        print("not found face..")
        return None

def predict_image(image):
    transform = transforms.Compose({
                    transforms.ToTensor()
                })
    image = transform(image)
    image = image.unsqueeze(dim=0)

    result = model(image)
    result = result.softmax(dim=1)
    return result

def judge_member(result):
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
    return name

def show_image(name):
    image = Image.open("./niziu_images/" + name + ".jpg")
    st.image(
            image, caption=name,
            use_column_width=True
        )

# 変数の定義
if 'results' not in st.session_state:
	st.session_state.results = torch.zeros((1, 9))

# レイアウト
# タイトル
st.title("Niziuのメンバー判定アプリ")
# 画像とボタンの配置
for i in range(1,6):
    print(i)
    col3, col4 = st.columns(2)

    with col3:
        image = opencv_resize_image("./images/0" + str(i) + ".jpg")
        if st.button(label="可愛いと思う", key=i):
            print("test")
            result = predict_image(image)
            st.session_state.results += result

        st.image(
            image, caption='1',
            use_column_width=True
        )


    with col4:
        image = opencv_resize_image("./images/0" + str(i + 5) + ".jpg")
        if st.button(label="可愛いと思う", key=i+5):
            result = predict_image(image)
            st.session_state.results += result

        st.image(
            image, caption='2',
            use_column_width=True
        )


if st.button("predict"):
    st.write("あなたが選んだメンバーは...")
    # st.write(st.session_state.results)
    name = judge_member(st.session_state.results)
    # st.write(name)
    show_image(name)

