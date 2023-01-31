# -*- coding: utf-8 -*-
import os
import cv2 
import time

emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

framepath = 'dataset/frames/'
videopath = 'dataset/videos/'
framenum = 0

if not os.path.exists(os.path.join(framepath)):
    os.mkdir(framepath)

#obtener todos los vídeos
video_list = os.listdir(videopath)

for i in range(1431,len(video_list)):
    print(f"Proceso: {i+1}/{len(video_list)} ", end="")
    tic = time.perf_counter()
    #el tercer argumento del vídeo muestra la emoción
    emo = int(video_list[i].split("-")[2]) - 1
    
    #creamos carpetas para las emociones
    if not os.path.exists(os.path.join(framepath, emotions[emo])):
        os.mkdir(framepath + emotions[emo])
       
    #leer el vídeo
    videocap = cv2.VideoCapture(os.path.join(*(videopath, video_list[i])))
    success,frame = videocap.read() 
    
    count = 0
    while success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detección del rostro
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in faces:
            sub_face = frame[y:y+h, x:x+w]
        
            #guardamos los fotogramas del vídeo leído en la carpeta de la clase a la que pertenece.
            facespath = os.path.join(*(framepath, emotions[emo])) + "/frame_" + str(framenum) + ".png"
            cv2.imwrite(facespath, sub_face)
            success,frame = videocap.read() 
            framenum += 1
            count += 1
    
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} segundos.")