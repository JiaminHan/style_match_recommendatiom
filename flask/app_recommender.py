#Usage: python app.py
import os
import glob
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import imutils
import time
import uuid
import base64
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from sklearn.metrics.pairwise import euclidean_distances
from keras.applications.xception import Xception
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import Sequential, Model,load_model,model_from_json
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import applications
from keras import optimizers
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors
import math
import operator
from sklearn.metrics.pairwise import cosine_distances
import pickle




global model
global model2
global df
global df_meta
global model_math
global feature_path
#img_width, img_height = 244, 244
# model_path = './models/vgg16_clothing_classifier_0911_v1.h5'
weights_path = './models/clothing_classifier.h5'
json_path= './models/clothing_classifier.json'
feature_path='./models/dress_features_new.pickle'
meta_path='./models/final_dress_meta.pickle'
#model_weights_path = './models/weights.h5'
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

    # load weights into new model
model.load_weights(weights_path)

# model = load_model(model_path)

model2 = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
df=pd.read_pickle(feature_path)
#cos=NearestNeighbors(n_neighbors=5,metric='cosine').fit(features)
#distances_cos, indices_cos=cos.kneighbors(features)
df_meta=pd.read_pickle(meta_path)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def image_preprocess(img_file):
    global img
    img = imread(img_file)
    img = resize(img, (224, 224), preserve_range=True).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict(file):
  x = image_preprocess(file)
  array = model.predict(x)
  res= array[0]
  answer= np.argmax(res)
  return answer

def find_alt(image_path):
    global a
    global av
    global dist
    global com
    global ids
    global recommend
    a=image_preprocess(image_path)
    # print(a)
    av = model2.predict(a)
    # print(av)
    dist = cosine_distances(av, np.array(df))[0]
    com=dict(zip(df.index,dist))
    ids=sorted(com, key=com.get)
    recommend=ids[:6]
    return recommend

def find_info(imgid):
    category=df_meta.loc[df_meta['id']==imgid,'parent_category'].values[0]
    img_path = df_meta.loc[df_meta['id']==imgid,'image_url'].values[0]
    #img_path='./data/train/dress/'+str(imgid)+'.jpg'
    price=df_meta.loc[df_meta['id']==imgid,'final_price'].values[0]
    url=df_meta.loc[df_meta['id']==imgid,'url'].values[0]
    name=df_meta.loc[df_meta['id']==imgid,'product_name'].values[0]
    return [img_path,price,url,name,category]


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")


def template_test():
    return render_template('template.html', label='', imagesource='./uploads/img5.jpg',
                           name1='', price1='', url1='',imagesource1='./uploads/img1.jpg',
                           name2='', price2='', url2='',imagesource2='./uploads/img2.jpg',
                           name3='', price3='', url3='',imagesource3='./uploads/img3.jpg',
                           name4='', price4='', url4='',imagesource4='./uploads/img4.jpg')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global recommendations
    global result1
    global result2
    global result3
    global result4

    if request.method == 'POST':
        import time

        start_time = time.time()
        file = request.files['file']

        #if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result = predict(file_path)
        global label
        if result == 0:
            label = 'Dress'
        elif result == 1:
            label = 'Dress'
        elif result == 2:
            label = 'Jeans'
        elif result == 3:
            label = 'Outerwear'
        elif result == 4:
            label = 'Pants'
        elif result == 5:
            label = 'Shorts'
        elif result == 6:
            label = 'Skirts'
        elif result == 7:
            label = 'Sweater'
        elif result == 8:
            label = 'Sweatshirt'
        elif result == 9:
            label = 'Tops'


        recommendations=find_alt(file_path)

        result1 = recommendations[0]
        result2 = recommendations[1]
        result3 = recommendations[2]
        result4 = recommendations[3]

        name1=find_info(result1)[3]
        price1=find_info(result1)[1]
        url1=find_info(result1)[2]
        imagesource1=find_info(result1)[0]

        name2 = find_info(result2)[3]
        price2 = find_info(result2)[1]
        url2 = find_info(result2)[2]
        imagesource2 = find_info(result2)[0]

        name3 = find_info(result3)[3]
        price3 = find_info(result3)[1]
        url3 = find_info(result3)[2]
        imagesource3 = find_info(result3)[0]

        name4 = find_info(result4)[3]
        price4 = find_info(result4)[1]
        url4 = find_info(result4)[2]
        imagesource4 = find_info(result4)[0]
        print(recommendations)

        # print(result1,result2,result3,result4)
        # print(file_path)
        filename = my_random_string(6) + filename

        os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("--- %s seconds ---" % str (time.time() - start_time))
        return render_template('template.html', label=label, imagesource='./uploads/' + filename,
                                   name1=name1,price1=price1, url1=url1,imagesource1=imagesource1,
                                   name2=name2,price2=price2, url2=url2, imagesource2=imagesource2,
                                   name3=name3,price3=price3, url3=url3,imagesource3=imagesource3,
                                   name4=name4,price4=price4, url4=url4,imagesource4=imagesource4)


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)