import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.

LR = 1e-3
TRAIN_DIR = 'vowel'
#MODEL_NAME = 'baybayinvowel-v1-{}-{}.model'.format(LEARNING_RATE, '2convlayers')
MODEL_NAME = 'baybayinvowel-v1-{}-{}.model'.format(LR, '2convlayers')
IMG_SIZE = 28
NUM_OUTPUT = 3


##START of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 16, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 16, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, NUM_OUTPUT, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
##END of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/


test_data = np.load(TRAIN_DIR+'_data.npy')
print('LOADING MODEL:', '{}.meta'.format(MODEL_NAME))
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('MODEL LOADED:', '{}.meta'.format(MODEL_NAME))

fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    #model_out_a = model.predict([data])[2]
    #model_out_e_i = model.predict([data])[1]
    #model_out_o_u = model.predict([data])[0]
    
    data_res = np.round(model.predict([data])[0], 0)
    print("RESULT:", data_res)
    str_label = 'NONE'
    if (data_res==[1,0,0]).all(): str_label='A'
    elif (data_res==[0,1,0]).all(): str_label='E / I'
    elif (data_res==[0,0,1]).all(): str_label='O / U'
    #str_label = model.predict([data])
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()