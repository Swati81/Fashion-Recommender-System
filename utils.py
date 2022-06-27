import pandas as pd
import requests
import cv2
from PIL import Image
import PIL
import pybase64
import numpy as np
import io
from numpy.linalg import norm
import tensorflow.keras as keras
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.densenet import DenseNet121,preprocess_input
import warnings
warnings.filterwarnings("ignore")

width, height, ch = 200, 200, 3
dense_net = DenseNet121(include_top=False, weights='imagenet', input_shape=(width, height, ch))
dense_net.trainable = False
# load densenet model
model = keras.Sequential([dense_net, GlobalMaxPooling2D()])


def Features():
    norm_emb = pd.read_csv('data/norm_embedings.csv')
    return np.array(norm_emb)


def embeding_image(image):
    im = cv2.resize(image, (200, 200), interpolation=cv2.INTER_NEAREST)
    x = np.expand_dims(im, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x).reshape(-1)
    res = pred/norm(pred)
    return np.array(res)


def get_indices(input_emb, features, neighbors):
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors([input_emb])
    return indices[::, 1:]


def recomendations(df, indices):
    # ids = []
    for i, ind in enumerate(indices[0]):
        Url = df.url[ind]
        iD = df.Id[ind]
        txt = f'ID:{str(iD)}'
        response = requests.get(Url)
        image_bytes = io.BytesIO(response.content)
        img = PIL.Image.open(image_bytes)
        img_arr = np.array(img)
        img_ar = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        height, width, ch = img_ar.shape
        ratio = width/480
        ht = int(height/ratio)
        img_ar = cv2.resize(img_ar, (480, ht))
        cv2.rectangle(img_ar, (0, 0), (110, 25), (255, 255, 255),-1)
        cv2.putText(img_ar, txt, (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # ids.append(txt)
        cv2.imwrite(f'static/{i+1}.jpg', img_ar)


def Decode(image):
    imgdata = pybase64.b64decode(image)
    image1 = np.asarray(bytearray(imgdata), dtype="uint8")
    image1 = cv2.imdecode(image1, cv2.IMREAD_COLOR)
    return image1
