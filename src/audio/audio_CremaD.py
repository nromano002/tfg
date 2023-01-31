# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy.random import seed

import os
import sys

import librosa
import librosa.display

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score,precision_score,recall_score

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

# main program starts here
if __name__ == '__main__':
    program = sys.argv[0]
    #argumentos: 1 (MFCC), 2(Chroma), 3(Mel), 4(STFT), 5(MFCC y Chroma), 6 (MFCC y Mel), 7(MFCC y STFT)
    if (len(sys.argv) == 3):
        modelo = sys.argv[1]
        feature = int(sys.argv[2])

seed(1)
tf.random.set_seed(2)       

if feature==1:
    featurestr='MFCC'
elif feature==2:
    featurestr='Chroma'
elif feature==3:
    featurestr='Mel'
elif feature==4:
    featurestr='STFT'
elif feature==5:
    featurestr='MFCC_Chroma'
elif feature==6:
    featurestr='MFCC_Mel'
elif feature==7:
    featurestr='MFCC_STFT'    
#Nombre de los archivos que se van a generar
archivo_txt= '../../resultados/audio/SER_CremaD_'+modelo+'_'+featurestr+'.txt'
archivo_h5 = '../../modelos/audio/SER_CremaD_'+modelo+'_'+featurestr+'.h5'


"""Dataset CremaD """
Crema = "../../inputs/cremad/AudioWAV/"

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
    

# dataframes para las emociones de los archivos
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframes para las rutas de los archivos
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()

labels = {'disgust':0,'happy':1,'sad':2,'neutral':3,'fear':4,'angry':5}
Crema_df.replace({'Emotions':labels},inplace=True)

#Extract features
def audio_features(signal, sample_rate, n_fft, hop_length,n_mfcc,feature):
    result = np.array([])
    #features
    mfcc=librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    chroma=librosa.feature.chroma_stft(signal, sample_rate, hop_length=hop_length)
    mel=librosa.feature.melspectrogram(signal, n_fft=n_fft, hop_length=hop_length)
    stft=librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    if feature==1:
        result=mfcc
    elif feature==2:
        result=chroma
    elif feature==3:
        result=librosa.feature.melspectrogram(signal, n_fft=n_fft, hop_length=hop_length)
    elif feature==4: #stft
        result=stft
    elif feature==5:
        result=np.concatenate((mfcc,chroma))
    elif feature==6:
        result=np.concatenate((mfcc,mel))
    else: 
        result=np.concatenate((mfcc,stft))
    return result

#get input shape for LSTM model
def get_input_shape(opt):
    if modelo=='LSTM' and opt==1: #lstm mfcc
        input_shape = (None,13)
    elif modelo!='LSTM' and opt==1:
        input_shape = (54,1,13)
        
    elif modelo=='LSTM' and opt==2: #lstm chroma
        input_shape = (None,12)
    elif modelo!='LSTM' and opt==2:
        input_shape = (54,1,12)
            
    elif modelo=='LSTM' and opt==3:  #lstm melspectogram
        input_shape = (None,128)
    elif modelo!='LSTM' and opt==3:
        input_shape = (54,1,128)
        
    elif modelo=='LSTM' and opt==4: #lstm stft
        input_shape=(None,2049)
    elif modelo!='LSTM' and opt==4:
        input_shape=(54,1,2049)
        
    elif modelo=='LSTM' and opt==5: 
       input_shape=(None,25) #lstm mfcc chroma
    elif modelo!='LSTM' and opt==5:   
       input_shape=(54,1,25)
       
    elif modelo=='LSTM' and opt==6: 
          input_shape=(None,141) #lstm #MFCC, mel
    elif modelo!='LSTM' and opt==6:    
         input_shape=(54,1,141)
         
    elif modelo=='LSTM' and opt==7: 
        input_shape=(None,2062) #lstm MFCC, STFT
    else:
        input_shape=(54,1,2062)
    return input_shape


# (33, 2062) mfcc stft
#Modelo LSTM
def build_LSTM_model():
    input_shape=get_input_shape(feature)
    model = tf.keras.Sequential()

    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(6, activation='softmax'))
    
    return model

def build_AlexNet_model():
    #Modelo
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=get_input_shape(feature), padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same"),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same"),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(6, activation='softmax')
    ])
    return model

def build_CNN_model():
    model=Sequential()
    #adding convolution layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=get_input_shape(feature),padding = 'same'))
    #adding pooling layer
    model.add(MaxPool2D(2,2,padding="same"))
    #adding fully connected layer
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    #adding output layer
    model.add(Dense(6,activation='softmax'))
    return model    

def build_CNN_LSTM_model():
    model = Sequential()
    model.add(Conv2D(16, (2,2), padding = 'same', input_shape=get_input_shape(feature)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),padding = 'same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (2,2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),padding = 'same'))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dense(6, activation='softmax'))
    return model

#crear modelo
def build_model():
    if modelo =='LSTM':
        return build_LSTM_model()
    elif modelo == 'AlexNet':
        return build_AlexNet_model()
    elif modelo == 'MLP': 
        return MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    elif modelo == 'CNN':
        return build_CNN_model()
    elif modelo == 'CNN_LSTM':
        return build_CNN_LSTM_model()
    

num_mfcc=13
n_fft=4096  #tamaño de la ventana FFT
hop_length=2048
SAMPLE_RATE = 22050
data = {
        "labels": [],
        "mfcc": []
    }
for i in range(7442):
    data['labels'].append(Crema_df.iloc[i,0])
    signal, sample_rate = librosa.load(Crema_df.iloc[i,1], sr=SAMPLE_RATE)
    #n_mfcc: nº de mfccs a devolver
    #hop_length: longitud de salto
    #tamaño de la ventana FFT
    mfcc=audio_features(signal, sample_rate,n_fft, hop_length, n_mfcc=13, feature=feature)
    mfcc = mfcc.T
    data["mfcc"].append(np.asarray(mfcc))


        
#igualar la longitud
X = np.asarray(data['mfcc'])
y = np.asarray(data["labels"])

X = tf.keras.preprocessing.sequence.pad_sequences(X)
print(X.shape)

#Stratified K fold (5/1)
accuracy=[]
skf= StratifiedKFold(n_splits=5,random_state=None)
skf.get_n_splits(X,y)
model=build_model()

if modelo=='MLP':
    end='true'
elif modelo!='CNN_LSTM':
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
else:
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
                  metrics=['accuracy'])
    


file= open(archivo_txt, 'a')
iter=1
cms=[]
F1s=[]
meanF1=[]
precisions=[]
recalls=[]
for train_index, test_index in skf.split(X,y):
    #file.write('Folder '+str(iter)+'\n')
    #file.write('Train\n{}\nTest\n{}\n\n'.format(train_index, test_index))
    print("Train", train_index, "Validation", test_index)
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    
    if modelo=='AlexNet' or modelo=='CNN' or modelo=='CNN_LSTM':
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
    if modelo!='MLP':
        history = model.fit(X_train, y_train, batch_size=32, epochs=30)
    if modelo=='MLP':
        nsamples, nx, ny = X_train.shape
        d2_train_dataset = X_train.reshape((nsamples,nx*ny))

        nsamples_test, nx_test, ny_test = X_test.shape
        d2_test_dataset = X_test.reshape((nsamples_test,nx_test*ny_test)) 

        model.fit(d2_train_dataset,y_train)
        # Predict for the test set
        y_pred=model.predict(d2_test_dataset)
        test_acc=accuracy_score(y_true=y_test, y_pred=y_pred)
        cr=classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4,5])
        F1=f1_score(y_test, y_pred, average='weighted')
        precision=precision_score(y_test, y_pred, average='weighted')
        recall=recall_score(y_test, y_pred, average='weighted')

    if modelo!='MLP':
        prediction = model.predict(X_test)
        pred=np.argmax(prediction, axis=1)
        cr=classification_report(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=[0,1,2,3,4,5])
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        #disp.plot()
        test_loss, test_acc = model.evaluate(X_test, y_test)
        F1=f1_score(y_test, pred, average='weighted')
        precision=precision_score(y_test, pred, average='weighted')
        recall=recall_score(y_test, pred, average='weighted')
    
    print(cr)
    print(cm)    
    print('Accuracy on folder',iter,'is:',test_acc*100)

    accuracy.append(test_acc)
    F1s.append(F1)
    cms.append(cm)
    precisions.append(precision)
    recalls.append(recall)
    
    
    file.write('Classification Report\n{}\nConfusion Matrix\n{}\n\n'.format(cr, cm)) 
    file.write('Test accuracy on folder '+str(iter)+'\n'+str((test_acc*100))+'\n\n')

    iter+=1
if modelo!='MLP':
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    model.save(archivo_h5)
              
model_accuracy= (np.array(accuracy).mean()) * 100
meanCM = np.mean(cms, axis=0).astype(np.int16)
meanF1= np.mean(F1s) * 100
meanPrecision=np.mean(precisions)  * 100
meanRecall= np.mean(recalls)  * 100

file.write('\nSUMMARY\n\n')
file.write('Mean accuracy\n{}\n\n'.format(model_accuracy))
file.write('F1-Score\n{}\n\n'.format(meanF1))
file.write('Precision\n{}\n\n'.format(meanPrecision))
file.write('Recall\n{}\n\n'.format(meanRecall))
file.write('Confusion Matrix\n{}'.format(meanCM))

print('SUMMARY')
print('Mean accuracy',model_accuracy)           
print('F1-Score',meanF1)
print('Precision',meanPrecision)
print('Recall',meanRecall)
print('Confusion Matrix', meanCM)
file.close()
