# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import os
import sys

import librosa
import librosa.display

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, MaxPool2D, Conv2D, MaxPooling2D, TimeDistributed,Activation
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold

from sklearn.neural_network import MLPClassifier

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import sys
# main program starts here
if __name__ == '__main__':
    program = sys.argv[0]
    #argumentos: 1 (MFCC), 2(Chroma), 3(Mel), 4(STFT), 5(MFCC y Chroma), 6 (MFCC y Mel), 7(MFCC y STFT)
    if (len(sys.argv) == 3):
        modelo = sys.argv[1]
        feature = int(sys.argv[2])

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

"""Dataset RAV """
RAV = "../../inputs/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

dir_list = os.listdir(RAV)
dir_list.sort()

emotion = []
gender = []
path = []
for i in dir_list:
    fname = os.listdir(RAV + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        path.append(RAV + i + '/' + f)
        


# dataframes para las emociones de los archivos
emotion_df = pd.DataFrame(emotion, columns = ['Emotions'])

# dataframes para las rutas de los archivos
path_df = pd.DataFrame(path, columns=['Path'])
RAV_df = pd.concat([emotion_df, path_df], axis=1)
def audio_features(signal, sample_rate, n_fft, hop_length,n_mfcc):
    result = np.array([])
    #features
    mfcc=librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    chroma=librosa.feature.chroma_stft(signal, sample_rate, hop_length=hop_length)
    mel=librosa.feature.melspectrogram(signal, n_fft=n_fft, hop_length=hop_length)
    stft=librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    result=np.concatenate((mfcc,stft))
    return result

num_mfcc=13
n_fft=4096  #tamaño de la ventana FFT
hop_length=2048
SAMPLE_RATE = 22050
data = {
        "labels": [],
        "mfcc": []
    }
for i in range(1440):
    data['labels'].append(RAV_df.iloc[i,0])
    signal, sample_rate = librosa.load(RAV_df.iloc[i,1], sr=SAMPLE_RATE)
    #n_mfcc: nº de mfccs a devolver
    #hop_length: longitud de salto
    #tamaño de la ventana FFT
    #mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfcc=audio_features(signal, sample_rate,n_fft, hop_length, n_mfcc=13)
    mfcc = mfcc.T
    data["mfcc"].append(np.asarray(mfcc))
    
X = np.asarray(data['mfcc'])
y = np.asarray(data["labels"])

X = tf.keras.preprocessing.sequence.pad_sequences(X)
print(X.shape)

X = np.array(X)
y=np.array(y)
np.save('../../inputs/X_audio_earlyfusion',X)
np.save('../../inputs/y_audio_earlyfusion',y)