# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 23:20:13 2020

@author: harsh
"""

import pymongo

import numpy as np
import pandas as pd

import cv2
import datetime

from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
import tensorflow as tf
import json

connection = pymongo.MongoClient('localhost',27017)

database = connection["supermarket"]

collection = database["entry_gate"]

with tf.device('/gpu:0'):
    detector = MTCNN()
    face_verification_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def get_embedding(frame, x1, y1, x2, y2, required_size = (224, 224)):
    tf.debugging.set_log_device_placement(True)
    with tf.device('/gpu:0'):
        face = frame[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
            
        samples = np.asarray([face_array], 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = face_verification_model.predict(samples)
            
    return yhat

def is_match(embedding, counter):
    tf.debugging.set_log_device_placement(True)
    with tf.device('/gpu:0'):
        
        if counter!=0:
            score = cosine(embedding, np.array(customers["P"+str(counter)]['embedding']))
            print("score: ", score)
            if score<=0.4:
                return True
            
            else:
                return False

tf.debugging.set_log_device_placement(True)
videoFile = "D:/Harsh/Mtech DS/Capstone Project/Supermarket/entry_gate_camera.mp4"
cap = cv2.VideoCapture(videoFile)   
frame_count = 0
counter = 0
customers = {}

while(cap.isOpened()):

    ret, frame = cap.read()
    
    if frame_count%5 == 0:
        
        if (ret == True):
            frame = cv2.resize(frame, ( 360*2, 240*2))

            with tf.device('/gpu:0'):
                results = detector.detect_faces(frame)
            #results = detector.detect_faces(frame)
            #cv2.rectangle(frame, (0,250), (1000, 250), (0, 255, 0), 2)
        
                if len(results)!=0:
                    print("Face detected")
                    
                    faces = np.array([i['box'] for i in results])
                    
                    faces[:,2] = faces[:,2] + faces[:, 0]
                    faces[:,3] = faces[:,3] + faces[:, 1]
                    faces = np.hstack((faces, np.reshape(((faces[:, 2]-faces[:, 0])*(faces[:,3]-faces[:,1])), (-1,1))))
                    for i in range(0, faces.shape[0]):
                        
                        cv2.rectangle(frame, (faces[i, 0], faces[i, 1]), (faces[i, 2], faces[i, 3]), (0, 255, 0), 2)
                        area = faces[i, 4]
                        print("area: ",area)
                        #print(3000<area<7000)
                        if (1500<area<2000):
                            print("Calculating id")
                            embedding = get_embedding(frame, faces[i,0], faces[i,1], faces[i,2], faces[i,3])
                            
                            if is_match(embedding, counter) != True:
                                counter = counter + 1
                                customers["P"+str(counter)] = {'embedding': embedding.tolist(), 'entry_time': str(datetime.datetime.now())}
                                print(embedding)
                                try_ = {"_id": "P"+str(counter), "embedding": embedding.tolist(), "entry_time": str(datetime.datetime.now())}
                                collection.insert_one(try_)
                            
                print("Counter: ", counter)
                frame = cv2.resize(frame, ( 700, 700))
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        else:
            break
                
    frame_count = frame_count + 1
 
cap.release()
cv2.destroyAllWindows()     

connection.close()
