# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pathlib

import sys
# main program starts here
if __name__ == '__main__':
    program = sys.argv[0]
    #argumentos: 1 modelo
    modelo = sys.argv[1]

data_path="../../inputs/dataset/frames/"
data__dir=pathlib.Path(data_path)
list(data__dir.glob('*.png'))[:2]


    
image_data_dic={
    'angry':list(data__dir.glob('angry/*')),
    'calm':list(data__dir.glob('calm/*')),
    'disgust':list(data__dir.glob('disgust/*')),
    'fearful':list(data__dir.glob('fearful/*')),
    'happy':list(data__dir.glob('happy/*')),
    'sad':list(data__dir.glob('sad/*')),
    'surprised':list(data__dir.glob('surprised/*')),
    'neutral':list(data__dir.glob('neutral/*')),
}
image_labels_dic={

    'neutral':1,
    'calm':2,
    'happy':3,
    'sad':4,
    'angry':5,
    'fearful':6,
    'disgust':7,
    'surprised':8,
}


X, y = [], []
i=0
for image_name, images in image_data_dic.items():
    for image in images:
        img = cv2.imread(str(image))
        if modelo=='fusion':
            resized_img = cv2.resize(img,(1,57))

        elif modelo!='VGG':
            resized_img = cv2.resize(img,(48,48))
        else:
            resized_img = cv2.resize(img,(75,75))
        X.append(resized_img) #X.append(resized_img)
        y.append(image_labels_dic[image_name])
    i=i+1
X = np.array(X)
y=np.array(y)
if modelo=='fusion':
    np.save('../../inputs/X_video_fusion',X)
    np.save('../../inputs/y_video_fusion',y)
elif modelo!='VGG':
    np.save('../../inputs/X_video',X)
    np.save('../../inputs/y_video',y)
else:
    np.save('../../inputs/X_video75',X)
    np.save('../../inputs/y_video75',y)
