# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 00:58:00 2020

@author: harsh
"""

import pymongo

import cv2
import datetime

import json
import dlib
from imutils import face_utils
import zmq

connection = pymongo.MongoClient('localhost',27017)

database = connection["supermarket"]

collection = database["internal_camera_tracking"]

face_detect = dlib.get_frontal_face_detector()
t= dlib.correlation_tracker()

context = zmq.Context()
sink = context.socket(zmq.PUSH)
sink.connect("tcp://localhost:8081")

def get_intersection(x1, y1, x2, y2, x1_, y1_, x2_, y2_):
    
    x5 = max(x1, x1_)
    y5 = max(y1, y1_)
    
    x6 = min(x2, x2_)
    y6 = min(y2, y2_)
    
    if (x5 > x6 or y5 > y6):
        return 0
    
    IntersationArea = (x6 - x5) * (y6 - y5)
    A1 = (x2 - x1) * (y2 - y1)
    A2 = (x2_ - x1_) * (y2_ - y1_)
    IOU = (IntersationArea / (A1 + A2 - IntersationArea))*100
    
    return IOU

def check_overlap(x1,y1,x2,y2):
    
   if counter!=0:
        
        for i in temp_customers.keys():
            
            x1_,y1_,x2_,y2_ = temp_customers[i]['current_location']
            
            IOU = get_intersection(x1,y1,x2,y2,x1_,y1_,x2_,y2_)
            
            if IOU >= 0.7:
                return (True, i)
        
   return (False, 0)

videoFile = "D:/Harsh/Mtech DS/Capstone Project/Supermarket/internal_camera.mp4"
cap = cv2.VideoCapture(videoFile)   

frame_count = 0
counter = 0
temp_customers = {}

temp = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if (ret == True):
        
        fram = cv2.resize(frame, ( 360*2, 240*2))
        frame = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
        
        if frame_count%10 == 0:
                        
            print("Detecting")
            
            results = face_detect(frame, 1)
            
            if len(results)!=0:
                print("Face detected")
                
                for faces in results:
                    (x1, y1, w, h) = face_utils.rect_to_bb(faces)
                    x2= x1 + w
                    y2= y1 + h
                                            
                    cond, fid = check_overlap(x1, y1, x2, y2)
                    print(cond)
                        
                    if cond == False:
                                                
                        
                        rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
                        print(rect)
                        t= dlib.correlation_tracker()
                        t.start_track(frame, rect)

                        counter = counter + 1
                        temp_customers[counter] = {"current_location": [x1, y1, x2, y2], "tracker": t}
                        
                        try_ = {str(datetime.datetime.now()): {"current_location": [x2-x1/2, y2-y1/2]}}
                        collection.update_one({"_id": "T"+str(counter)}, {"$set": try_}, upsert=True)
                    
                        #send(counter, frame, x1,y1,x2,y2)
                        
                        dict_to_send = json.dumps({"fid": "T"+str(counter), "frame": fram.tolist(), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                        sink.send_json(dict_to_send)
                        
                        cv2.rectangle(fram, (x1, y1), (x2, y2), (0, 0, 0), 3)

        else:
            
            print("tracking")                
            rem_list = []
            #tr_rem_list = []
            
            for i in temp_customers.keys():
                
                print("Tracking: ", i)
                #if customers[i]["tracker"] != None:
                t = temp_customers[i]["tracker"]
                quality = t.update(frame)
                print(i, " = ", quality)
                if quality > 4:
                                        
                    pos = t.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                
                    temp_customers[i]['current_location'] = [startX, startY, endX, endY] 
                    
                    try_ = {str(datetime.datetime.now()): {"current_location": [x2-x1/2, y2-y1/2]}}
                    collection.update_one({"_id": "T"+str(counter)}, {"$set": try_}, upsert=True)
                    
                    temp = temp + 1
                    print(endX-startX/2, endY-startY/2)
                    cv2.rectangle(fram, (startX, startY), (endX, endY),(255, 255, 255), 2)
                    
                else:
                    rem_list.append(i)
                    
                
            #for i in tr_rem_list:
             #   customers[i]["tracker"] = None
            
            [temp_customers.pop(key) for key in rem_list] 
                    
        #cv2.rectangle(fram, (0,400), (500,400), (0, 255, 0), 2)           
        disp_frame = cv2.resize(fram, ( 700, 700))
        cv2.imshow("Video", disp_frame)
        
        frame_count= frame_count + 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        
        print(len(temp_customers))
    else:
        break

        
cap.release()
cv2.destroyAllWindows()

connection.close()