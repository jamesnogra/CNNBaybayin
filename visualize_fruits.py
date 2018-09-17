import matplotlib.pyplot as plt
import numpy as np
import os
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.

LEARNING_RATE = 1e-3
NUM_OUTPUT = 4
#MODEL_NAME = 'baybayinvowel-v1-{}-{}.model'.format(LEARNING_RATE, '2convlayers')
MODEL_NAME = 'orangeapple-{}-{}.model'.format(LEARNING_RATE, '2conv-basic') # just so we remember which saved model is which, sizes must match
IMG_SIZE = 50


##START of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, NUM_OUTPUT, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LEARNING_RATE, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
##END of tflearn CNN. From: https://pythonprogramming.net/tflearn-machine-learning-tutorial/


test_data = np.load('train_data.npy')
print('LOADING MODEL:', '{}.meta'.format(MODEL_NAME))
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

fig=plt.figure()

for num,data in enumerate(test_data[:40]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(5,8,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    #model_out_a = model.predict([data])[2]
    #model_out_e_i = model.predict([data])[1]
    #model_out_o_u = model.predict([data])[0]
    
    data_res = np.round(model.predict([data])[0], 0)
    print("RESULT:", data_res)
    if (data_res==[0.,1.,0.,0.]).all(): str_label='Grapes'
    if (data_res==[0.,0.,1.,0.]).all(): str_label='Oranges'
    if (data_res==[0.,0.,0.,1.]).all(): str_label='Apples'
    #if np.argmax(model_out_o_u) == 1: str_label='O/U'
    #else: str_label='A or E/I'
    #str_label = model.predict([data])
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()