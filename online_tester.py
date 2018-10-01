from flask import Flask, request, Response
from flask.json import jsonify
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys

LR = 1e-4
TRAIN_DIR = 'train_jpg'
#MODEL_NAME = 'baybayinvowel-v1-{}-{}.model'.format(LEARNING_RATE, '2convlayers')
MODEL_NAME = 'baybayin-v2-{}-{}.model'.format(LR, '4convlayers')
IMG_SIZE = 28
NUM_OUTPUT = 59

FILTER_SIZE = 3
NUM_EPOCHS = 50
FIRST_NUM_CHANNEL = 32


##START of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, FIRST_NUM_CHANNEL, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*2, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*4, FILTER_SIZE, activation='relu')
convnet = max_pool_2d(convnet, 2)

#convnet = conv_2d(convnet, FIRST_NUM_CHANNEL*8, FILTER_SIZE, activation='relu')
#convnet = max_pool_2d(convnet, FILTER_SIZE)

convnet = fully_connected(convnet, FIRST_NUM_CHANNEL*8, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, NUM_OUTPUT, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
##END of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/


print('LOADING MODEL:', '{}.meta'.format(MODEL_NAME))
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Model',MODEL_NAME,'loaded...')

def classifier(np_img):
	data = np_img.reshape(IMG_SIZE,IMG_SIZE,1)
	data_res_float = model.predict([data])[0]
	data_res = np.round(data_res_float, 0)

app = Flask(__name__)

@app.route('/test-upload')
def testUpload():
	return '<form action="/classify-image" method="post" enctype="multipart/form-data"><input type="file" name="the_image" /><button type="submit">Upload</button></form>'

@app.route('/classify-image', methods=['POST'])
def classifyImage():
	try:
		img = cv2.imdecode(np.fromstring(request.files['the_image'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
		img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] # convert image to black and white pixels
		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		classifier(img)
		#print(img)
		#return jsonify(display_array)
		return jsonify({'status': 1, 'message': 'Image classification complete.', 'result': {}})
	except:
		return jsonify({'status': -1, 'message': 'Probably not an image!'})

@app.route('/')
def hello():
	return "Hello World!"

if __name__ == '__main__':
	app.run(debug=True, port=8080)