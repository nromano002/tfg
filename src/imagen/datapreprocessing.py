# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from tensorflow.keras.utils import to_categorical
"""FER-2013 """
data = pd.read_csv('../../inputs/fer2013.csv')
data.shape
data.Usage.value_counts()

emotion_map = {0: ' Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)

x = []
y = []
first = True
for line in open("../../inputs/fer2013.csv"):
    if first:
        first = False
    else:
        row = line.split(',')
        x.append([int(p) for p in row[1].split()])
        y.append(int(row[0]))
X, y = np.array(x) / 255.0, np.array(y)

X = X.reshape(-1, 48, 48, 1)

np.save('../../inputs/X_image',X)
np.save('../../inputs/y_image',y)

img_height, img_width = 139, 139
# Preprocesses a numpy array encoding a batch of images
    # x: Input array to preprocess
def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


def get_data():
    pixels = data['pixels'].tolist()
    images = np.empty((len(data), img_height, img_width, 1))
    i = 0

    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence.split(' ')]  # Extraction of each single
        single_image = np.asarray(single_image).reshape(48, 48) # Dimension: 48x48
        single_image = resize(single_image, (img_height, img_width), order = 1, mode = 'constant') # Dimension: 139x139x3 (Bicubic)
        ret = np.empty((img_height, img_width, 1))  
        ret[:, :, 0] = single_image
        images[i, :] = ret
        i += 1
    
    X_Inceptionv3 = preprocess_input(images)
    y_Inceptionv3 = data['emotion']

    return X_Inceptionv3, y_Inceptionv3    
X_Inceptionv3,y_Inceptionv3 =get_data()
np.save('../../inputs/X_Inceptionv3_image',X_Inceptionv3)
np.save('../../inputs/y_Inceptionv3_image',y_Inceptionv3)