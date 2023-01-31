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
from tensorflow.keras.layers import Dense, LSTM, AveragePooling2D, Input,Add,GlobalAveragePooling2D, Flatten, Dropout, MaxPool2D, Conv2D, MaxPooling2D,BatchNormalization, TimeDistributed,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop, SGD

from keras.initializers import glorot_uniform

from sklearn.model_selection import StratifiedKFold

from sklearn.neural_network import MLPClassifier

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3



import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# main program starts here
if __name__ == '__main__':
    program = sys.argv[0]
    #argumentos: 1 modelo
    modelo = sys.argv[1]

seed(1)
tf.random.set_seed(2)


#Nombre de los archivos que se van a generar
archivo_txt= '../../resultados/imagen/IER_'+modelo+'.txt'
archivo_h5 = '../../modelos/imagen/IER_'+modelo+'.h5'

"""FER-2013 """
if modelo!='Inceptionv3':
    X = np.load('../../inputs/X_image.npy')
    y = np.load('../../inputs/y_image.npy')
else:
    X = np.load('../../inputs/X_Inceptionv3_image.npy')
    y = np.load('../../inputs/y_Inceptionv3_image.npy')

def build_Inceptionv3_model():
    base_model = InceptionV3(include_top = False, weights= 'imagenet',input_shape = (139, 139, 3))
    # Freezing Layers
    for layer in base_model.layers[:-4]:
        layer.trainable=False
    model = Sequential([
      tf.keras.layers.Lambda(tf.image.grayscale_to_rgb),
      base_model])
    model.add(GlobalAveragePooling2D())
    model.add(Dense(7,activation='softmax'))
    return model

def build_VGG16_model():
    base_model = tf.keras.applications.VGG16(input_shape=(48,48,3),include_top=False,weights="imagenet")
    
    # Freezing Layers
    for layer in base_model.layers[:-4]:
        layer.trainable=False
    model = Sequential([
      tf.keras.layers.Lambda(tf.image.grayscale_to_rgb),
      base_model])
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(32,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(7,activation='softmax'))
    return model



def build_LSTM_model():
    model = Sequential()
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dense(7, activation='softmax'))
    return model

def build_AlexNet_model():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='same',
                     input_shape=(48,48,1)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    return model

def build_DCNN_model():
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal', padding='same',input_shape = (48,48,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2),padding='same'))
    model.add(Dropout(0.3))


    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2),padding='same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 256, kernel_size = (3,3), activation = 'elu', kernel_initializer='he_normal',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2),padding='same'))
    model.add(Dropout(0.5))


    model.add(Flatten())

    model.add(Dense(units = 1024, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units = 512, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units = 256, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(Dense(units = 7, activation = 'softmax'))
    return model

def identity_block(X1, f, filters, stage, block):
    """
    Implementation of the identity block
    
    Arguments:
    X1 -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X1 -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X1
    
    # First component of main path
    X1 = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X1)
    X1 = Activation('relu')(X1)
    
   
    # Second component of main path
    X1 = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X1)
    X1 = Activation('relu')(X1)

    # Third component of main path 
    X1 = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X1)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X1 = Add()([X_shortcut,X1])
    X1 = Activation("relu")(X1)
        
    return X1

def convolutional_block(X1, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X1


    ##### MAIN PATH #####
    # First component of main path 
    X1 = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X1)
    X1 = Activation('relu')(X1)
    
    # Second component of main path 
    X1 = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b',padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X1)
    X1 = Activation('relu')(X1)

    # Third component of main path 
    X1 = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X1)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X1 = Add()([X_shortcut,X1])
    X1 = Activation("relu")(X1)     
    return X1


#mezclar los dos bloques
def build_ResNet50_model(input_shape = (48,48,1), classes = 7):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    #X = ZeroPadding2D((1, 1))(X_input)
    X1 = X_input
    # Stage 1

    X1 = Conv2D(8, (3, 3), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = BatchNormalization(axis = 3, name = 'bn_conv1')(X1)
    X1 = Activation('relu')(X1)
    # removed maxpool
    #X = MaxPooling2D((3, 3), strides=(2, 2))(X1)

    # Stage 2
    X1 = convolutional_block(X1, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)
    X1 = identity_block(X1, 3, [32, 32, 128], stage=2, block='b')
    X1 = identity_block(X1, 3, [32, 32, 128], stage=2, block='c')


    # Stage 3 
    X1 = convolutional_block(X1, f = 3, filters = [64,64,256], stage = 3, block='a', s = 2)
    X1 = identity_block(X1, 3, [64,64,256], stage=3, block='b')
    X1 = identity_block(X1, 3, [64,64,256], stage=3, block='c')
    X1 = identity_block(X1, 3, [64,64,256], stage=3, block='d')

    # Stage 4 
    X1 = convolutional_block(X1, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)
    X1 = identity_block(X1, 3, [128, 128, 512], stage=4, block='b')
    X1 = identity_block(X1, 3, [128, 128, 512], stage=4, block='c')
    X1 = identity_block(X1, 3, [128, 128, 512], stage=4, block='d')
    X1 = identity_block(X1, 3, [128, 128, 512], stage=4, block='e')
    X1 = identity_block(X1, 3, [128, 128, 512], stage=4, block='f')

    # Stage 5 
    X1 = convolutional_block(X1, f = 3, filters = [256, 256, 1024], stage = 5, block='a', s = 2)
    X1 = identity_block(X1, 3, [256, 256, 1024], stage=5, block='b')
    X1 = identity_block(X1, 3, [256, 256, 1024], stage=5, block='c')

    # AVGPOOL . 
    X1 = AveragePooling2D((2,2), name='avg_pool')(X1)
    
    # output layer
    X1 = Flatten()(X1)
    X1 = Dense(512, activation = 'relu', name='fc1024' , kernel_initializer = glorot_uniform(seed=0))(X1)
    X1 = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X1)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X1, name='Net50')

    return model

def build_VGG19_model():
    base_model = tf.keras.applications.VGG19(input_shape=(48,48,3),include_top=False,weights="imagenet")
    # Freezing Layers
    for layer in base_model.layers[:-4]:
        layer.trainable=False
    model = Sequential([
      tf.keras.layers.Lambda(tf.image.grayscale_to_rgb),
      base_model])
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(32,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32,kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(7,activation='softmax'))
    return model

def build_CNN_model():
    model=Sequential()
    #adding convolution layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)))
    #adding pooling layer
    model.add(MaxPool2D(2,2))
    #adding fully connected layer
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    #adding output layer
    model.add(Dense(7,activation='softmax'))
    return model

def build_CNN_LSTM_model():
    model = Sequential()
    model.add(Conv2D(16, (2,2), padding = 'same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (2,2), padding = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dense(7, activation='softmax'))
    return model

#crear modelo
def build_model():
    if modelo == 'VGG16':
        return build_VGG16_model()
    elif modelo == 'VGG19':
        return build_VGG19_model()
    elif modelo =='LSTM':
        return build_LSTM_model()
    elif modelo =='AlexNet':
        return build_AlexNet_model()
    elif modelo =='DCNN':
        return build_DCNN_model()
    elif modelo =='CNN':
        return build_CNN_model()
    elif modelo =='CNN_LSTM':
        return build_CNN_LSTM_model()
    elif modelo=='ResNet50':
        return build_ResNet50_model(input_shape = (48, 48, 1), classes = 7)
    elif modelo=='Inceptionv3':
        return build_Inceptionv3_model()

#Stratified K fold (5/1)
accuracy=[]
skf= StratifiedKFold(n_splits=5,random_state=None)
skf.get_n_splits(X,y)

if modelo=='VGG16':
    model=build_VGG16_model()
else:
    model=build_model()

if modelo=='VGG16' or modelo=='AlexNet' or modelo=='VGG19' or modelo=='Inceptionv3' or modelo=='CNN_LSTM' or modelo=='ResNet50' or modelo=='LSTM':
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

else:
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

file= open(archivo_txt, 'a')
iter=1
cms=[]
F1s=[]
meanF1=[]
precisions=[]
recalls=[]
for train_index, test_index in skf.split(X,y):
    file.write('Folder '+str(iter)+'\n')
    file.write('Train\n{}\nTest\n{}\n\n'.format(train_index, test_index))
    print("Train", train_index, "Validation", test_index)
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
    num_class = len(set(y))
    if modelo=='Inceptionv3':
      history = model.fit(X_train, y_train, batch_size=64, epochs=75)
    elif modelo=='VGG16':
        history = model.fit(X_train, y_train,batch_size=64, epochs=30)
    elif modelo=='LSTM':
        history = model.fit(X_train, y_train,batch_size=64, epochs=50)
    elif modelo=='CNN_LSTM':
        history = model.fit(X_train, y_train, batch_size=32, epochs= 150)
    elif modelo=='CNN' or modelo=='VGG19':
        history = model.fit(X_train, y_train, batch_size=32, epochs= 75)	
    else:            
        history = model.fit(X_train, y_train, batch_size=32, epochs= 30)
     
    prediction=model.predict(X_test) 
    y_pred=np.argmax(prediction,axis=1)
        
    test_loss, test_acc = model.evaluate(X_test, y_test)
    

    cr=classification_report(y_test, y_pred)
    cm=confusion_matrix(y_test,y_pred)
    F1=f1_score(y_test, y_pred, average='weighted')
    precision=precision_score(y_test, y_pred, average='weighted')
    recall=recall_score(y_test, y_pred, average='weighted')

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
