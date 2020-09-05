# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 01:36:51 2020

@author: harsh
"""

import numpy as np

import pymongo
from PIL import Image

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
import tensorflow as tf

physical_devices= tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import json
import zmq

connection = pymongo.MongoClient('localhost',27017)

database = connection["supermarket"]

collection = database["internal_camera_recognition"]
entry_collection = database["entry_gate"]

face_verification_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:8081")


def get_embedding(frame, x1, y1, x2, y2, required_size = (224, 224)):
   
    face = frame[y1:y2, x1:x2]
    image = Image.fromarray((face).astype(np.uint8))
    image = image.resize(required_size)
    face_array = np.asarray(image)
            
    samples = np.asarray([face_array], 'float32')
    samples = preprocess_input(samples, version=2)
    yhat = face_verification_model.predict(samples)
    print("embedding created")
            
    return yhat

def is_match(embedding):
    
    customers_entry = entry_collection.find()
    
        
    for i in customers_entry:
        score = cosine(embedding, np.array(i['embedding']))
        print(score)
        if score<=0.52:
            print("Match Found", i["_id"])
            
            return i["_id"]
        
    return 0
    
def recognize(fid, frame, x1, y1, x2, y2):
    
    frame = np.asarray(frame)
    embedding = get_embedding(frame, x1, y1, x2, y2)
    
    person = is_match(embedding)
    
    try_ ={"permanent_id": person}
    collection.update_one({"_id": fid}, {"$set": try_}, upsert=True)
    
    print(person)

counter = 0

while True:
    print("listening")

    re = json.loads(receiver.recv_json())
    
    if len(re)!=0:
        print("message recieved")
        counter = counter + 1
        recognize(re['fid'], re['frame'], re['x1'], re['y1'], re['x2'], re['y2'])
    
    
connecttion.close()