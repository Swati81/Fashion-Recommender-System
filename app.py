from flask import Flask, render_template, request
from flask_cors import cross_origin
import cv2
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from utils import embeding_image, get_indices, recomendations, Features, Decode

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')

# load info file
df = pd.read_csv('data/info.csv')

# load features of fashion images
features = Features()


app = Flask(__name__)


@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
@cross_origin()
def result():
    if request.method == 'POST':
        image = request.json['image']
        img = Decode(image).copy()
        cv2.imwrite('static/input.jpg', img)
        input_emb = embeding_image(img)
        indices = get_indices(input_emb, features, neighbors)
        recomendations(df, indices)
        return render_template('index.html')
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
