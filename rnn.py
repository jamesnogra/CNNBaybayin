import tensorflow as tf
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.

TRAIN_DIR = 'train_jpg'
IMG_SIZE = 28
LR = 1e-4
LR_DECAY = 1e-6
NUM_OUTPUT = 64
NUM_EPOCHS = 50
DROPOUT = 0.2

def label_character(img):
    word_label = img.split('.')[0]
    #print("LABEL:", word_label)
    if word_label == 'a':         return 1
    elif word_label == 'e_i':     return 2
    elif word_label == 'o_u':     return 3
    elif word_label == 'ba':      return 4
    elif word_label == 'be_bi':   return 5
    elif word_label == 'bo_bu':   return 6
    elif word_label == 'b':       return 7
    elif word_label == 'ka':      return 8
    elif word_label == 'ke_ki':   return 9
    elif word_label == 'ko_ku':   return 10
    elif word_label == 'k':       return 11
    elif word_label == 'da_ra':   return 12
    elif word_label == 'de_di':   return 13
    elif word_label == 'do_du':   return 14
    elif word_label == 'd':       return 15
    elif word_label == 'ga':      return 16
    elif word_label == 'ge_gi':   return 17
    elif word_label == 'go_gu':   return 18
    elif word_label == 'g':       return 19
    elif word_label == 'ha':      return 20
    elif word_label == 'he_hi':   return 21
    elif word_label == 'ho_hu':   return 22
    elif word_label == 'h':       return 23
    elif word_label == 'la':      return 24
    elif word_label == 'le_li':   return 25
    elif word_label == 'lo_lu':   return 26
    elif word_label == 'l':       return 27
    elif word_label == 'ma':      return 28
    elif word_label == 'me_mi':   return 29
    elif word_label == 'mo_mu':   return 30
    elif word_label == 'm':       return 31
    elif word_label == 'na':      return 32
    elif word_label == 'ne_ni':   return 33
    elif word_label == 'no_nu':   return 34
    elif word_label == 'n':       return 35
    elif word_label == 'nga':     return 36
    elif word_label == 'nge_ngi': return 37
    elif word_label == 'ngo_ngu': return 38
    elif word_label == 'ng':      return 39
    elif word_label == 'pa':      return 40
    elif word_label == 'pe_pi':   return 41
    elif word_label == 'po_pu':   return 42
    elif word_label == 'p':       return 43
    elif word_label == 'sa':      return 44
    elif word_label == 'se_si':   return 45
    elif word_label == 'so_su':   return 46
    elif word_label == 's':       return 47
    elif word_label == 'ta':      return 48
    elif word_label == 'te_ti':   return 49
    elif word_label == 'to_tu':   return 50
    elif word_label == 't':       return 51
    elif word_label == 'wa':      return 52
    elif word_label == 'we_wi':   return 53
    elif word_label == 'wo_wu':   return 54
    elif word_label == 'w':       return 55
    elif word_label == 'ya':      return 56
    elif word_label == 'ye_yi':   return 57
    elif word_label == 'yo_yu':   return 58
    elif word_label == 'y':       return 59
    elif word_label == 'ra':      return 60
    elif word_label == 're_ri':   return 61
    elif word_label == 'ro_ru':   return 62
    elif word_label == 'r':       return 63
    print("UNLABEL (will cause an error):", img)

def create_train_data():
    training_data = []
    labeled_data = []
    all_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_character(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] # convert image to black and white pixels
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        all_data.append([img, label])
    shuffle(all_data)
    for item in tqdm(all_data):
        training_data.append(np.array(item[0]))
        labeled_data.append(np.array(item[1]))
    return np.array(training_data), np.array(labeled_data)

train_data, output_data = create_train_data()
#divide the data for training and validation
train_data = train_data / 255
train_x = train_data[:8500]
train_y = output_data[:8500]
validation_x = train_data[-1200:]
validation_y = output_data[-1200:]

#MNIST
#mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
#(train_x, train_y),(validation_x, validation_y) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test
#train_x = train_x/255.0
#validation_x = validation_x/255.0


#MODEL of LSTM
model = Sequential()

model.add(LSTM(64, input_shape=(train_x.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(DROPOUT))

model.add(LSTM(32, activation='relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(16, activation='relu'))
model.add(Dropout(DROPOUT))

model.add(Dense(NUM_OUTPUT, activation='softmax'))
#END of MODEL of LSTM


opt = tf.keras.optimizers.Adam(lr=LR, decay=LR_DECAY)


# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(train_x,
          train_y,
          epochs=NUM_EPOCHS,
          validation_data=(validation_x, validation_y))