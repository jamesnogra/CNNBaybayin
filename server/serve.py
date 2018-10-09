from flask import Flask, request, Response
from flask.json import jsonify
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys
import json
from operator import itemgetter
from flask_cors import CORS #pip install -U flask-cors
import base64

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-4
TRAIN_DIR = 'train_jpg'
#MODEL_NAME = 'baybayinvowel-v1-{}-{}.model'.format(LEARNING_RATE, '2convlayers')
MODEL_NAME = 'baybayin-v2-{}-{}.model'.format(LR, '4convlayers')
IMG_SIZE = 28
NUM_OUTPUT = 63

FILTER_SIZE = 3
NUM_EPOCHS = 50
FIRST_NUM_CHANNEL = 32

all_chars = ['A', 'E/I', 'O/U', 'BA', 'BE/BI', 'BO/BU', 'B', 'KA', 'KE/KI', 'KO/KU', 'K', 'DA', 'DE/DI', 'DO/DU', 'D', 'GA', 'GE/GI', 'GO/GU', 'G', 'HA', 'HE/HI', 'HO/HU', 'H', 'LA', 'LE/LI', 'LO/LU', 'L', 'MA', 'ME/MI', 'MO/MU', 'M', 'NA', 'NE/NI', 'NO/NU', 'N', 'NGA', 'NGE/NGI', 'NGO/NGU', 'NG', 'PA', 'PE/PI', 'PO/PU', 'P', 'SA', 'SE/SI', 'SO/SU', 'S', 'TA', 'TE/TI', 'TO/TU', 'T', 'WA', 'WE/WI', 'WO/WU', 'W', 'YA', 'YE/YI', 'YO/YU', 'Y', 'RA', 'RE/RI', 'RO/RU', 'R']
all_chars.reverse() #revese the all_chars array because it has been encoded here in reverse

def get_model_api():
	"""Returns lambda function for api"""
	# 1. initialize model once and for all and reload weights
	if os.path.exists('{}.meta'.format(MODEL_NAME)):
		##START of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
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
		result_chars = []
		model.load(MODEL_NAME)

	def model_api(np_img):
		data = np_img.reshape(IMG_SIZE,IMG_SIZE,1)
		data_res_float = model.predict([data])[0]
		data_res = np.round(data_res_float, 0)
		for x in range(len(data_res)):
			result_chars.append([all_chars[x], round((data_res_float[x]*100),4), data_res_float[x]])
		result_chars = sorted(result_chars, key=itemgetter(2)) #sort by the res_float
		print('Result:',result_chars[NUM_OUTPUT-1][0])
		return data_res_float, data_res, result_chars, result_chars[NUM_OUTPUT-1][0] #the last element is the correct classification

	return model_api