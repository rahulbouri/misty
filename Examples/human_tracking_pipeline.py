from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import os
import sys
import base64
import math
import time
import numpy as np
from transformers import pipeline
from PIL import Image
import cv2 as cv
import open3d as o3d
import plyfile
import requests


sys.path.insert(0, "/home/rahul/Desktop/misty_github/")


global yaw_value

def imu_callback(data):
    global yaw_value 

    if 'message' in data:
        imu_data = data['message']

        yaw_value = imu_data.get('yaw')

        return yaw_value
    else    :
        print("IMU data format is not as expected.")

        return None
    
global obstacle, location, hazard

def tof_call(data):

    global obstacle, location, hazard
    if 'message' in data:
        message = data['message']
        # print("\n-----------------\n")
        # print("distance in meters", message.get('distanceInMeters'))
        # print("in hazard:", message.get('inHazard'))
        # print("Sensor ID", message.get('sensorId'))
        # print("Sensor Position: ", message.get('sensorPosition'))
        # print("Status: ", message.get('status'))
        # print("Type: ", message.get('type'))
        # print("\n-----------------\n")

        if (message.get('status')==0 or message.get('status')==100):
            left = message.get('distanceInMeters') if message.get('sensorId') == 'tofdfl' else np.inf
            right = message.get('distanceInMeters') if message.get('sensorId') == 'tofdfr' else np.inf
            centre = message.get('distanceInMeters') if message.get('sensorId') == 'toffc' else np.inf
            obstacle = min(left, right, centre)

            if obstacle == left:
                location = 'left'
            elif obstacle == right:
                location = 'right'
            else:
                location = 'centre'

            # if message.get('sensorID') in ['toffr', 'toffl', 'toffc', 'tofdfr', 'tofdfl']:
            #     obstacle = message.get('distanceInMeters')
            #     hazard = message.get('inHazard')
            #     location = 'front'
            # else:
            #     location = 'back'
            # else:
            #     location = 'back'
        
        else:
            obstacle = np.inf
            location = 'centre'
    else:
        print("Data format is not as expected.")

def calculate_absolute_heading(current_heading, heading_difference):
    # Calculate the absolute heading using the 0 to 360 degrees representation.
    absolute_heading = (int(current_heading) + int(heading_difference)) % 360.0

    # Convert to the -180 to 180 degrees representation if necessary.
    if absolute_heading > 180:
        absolute_heading -= 360

    return absolute_heading

class ObjectDetection:
    def __init__(self, ip):
        self.ip = ip
        self.checkpoint = "vinvino02/glpn-nyu"
        self.depth_estimator = pipeline("depth-estimation", model = self.checkpoint)
        self.misty = Robot(self.ip)
        self.misty.set_default_volume(volume=30)
        self.yaw = None

    def yolo_object_detection(self, img_file='/home/rahul/Desktop/misty_github/Examples/test.jpg', goal='person'):

        labels_path = '/home/rahul/Desktop/misty_github/yolo-coco/coco.names'
        labels = open(labels_path).read().strip().split("\n")

        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

        weights_path = '/home/rahul/Desktop/misty_github/yolo-coco/yolov3.weights'
        config_path = '/home/rahul/Desktop/misty_github/yolo-coco/yolov3.cfg'

        net = cv.dnn.readNetFromDarknet(config_path, weights_path)

        image = cv.imread(img_file)
        (H, W) = image.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]

        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:

            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.8 and labels[classID] == goal:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        if len(idxs) > 0:
            
            biggest_box = 0

            rect_vertices = None
            goal_detected = False
            
            for i in idxs.flatten():
            
                (x,y) = (boxes[i][0], boxes[i][1])
                (w,h) = (boxes[i][2], boxes[i][3])
                
                color = [int(c) for c in colors[classIDs[i]]]

                cv.rectangle(image, (x, y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv.putText(image, text, (x,y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if labels[classIDs[i]] == goal:
                    # coordinates = [(x,y), (x+w, y), (x, y+h), (x+w, y+h)]
                    rect_vertices = x, y, w , h
                    goal_detected = True
                    break
                elif math.sqrt(w**2 + h**2) > biggest_box:
                    biggest_box = math.sqrt(w**2 + h**2)
                    # coordinates = [(x,y), (x+w, y), (x, y+h), (x+w, y+h)]
                    rect_vertices = x, y, w, h
                else:
                    rect_vertices = None

        # cv.imshow("Image", image)
        # cv.waitKey(0) 
        cv.imwrite('/home/rahul/Desktop/misty_github/Examples/test_yolo.jpg', image)       

        return rect_vertices, goal_detected
    
    def get_distance_image(self, goal='person'):
        count = 0 
        while True:
            url = "http://"+self.ip+"/api/cameras/depth"
            response = requests.get(url)
            parsed_data = response.json()
            height = parsed_data['result']['height']
            dist_image = parsed_data['result']['image']
            width = parsed_data['result']['width']
            status = parsed_data['status']

            self.take_image()

            dist_image = np.array(dist_image, dtype=np.float32)
            dist_image = np.array(dist_image).reshape(height, width)

            try:
                rect_vertices, _ = self.yolo_object_detection('/home/rahul/Desktop/misty_github/Examples/test.jpg', goal)
                x, y, w, h = rect_vertices
                roi = dist_image[x:x + w, y:y + h]
                valid_values = roi[~np.isnan(roi)]

                if len(valid_values) > 0:
                    distance = np.min(valid_values)
                    # break
                else:
                    distance  = np.nan
                
                if distance != np.nan:
                    rotate = False
                    break
                
            except UnboundLocalError as e:
                distance = 0
                rotate = True
                break
                    
       
        distance = (distance/1000)
        print("Distance to object is ", distance)
        return distance, rotate 
    
    def take_image(self):
        self.misty.take_picture(base64=True, fileName="test", width=320, height=240, displayOnScreen=False, overwriteExisting=True)
        img_b64 = self.misty.get_image(fileName="test.jpg", base64=True).json()['result']['base64']
        image_data = base64.b64decode(img_b64)
        directory = '/home/rahul/Desktop/misty_github/Examples'
        filename = 'test.jpg'
        with open(os.path.join(directory, filename), "wb") as image_file:
            image_file.write(image_data)
    
    def get_current_yaw(self):
        IMU = self.misty.register_event("IMU", Events.IMU, condition=None, debounce=0, keep_alive=False, callback_function=imu_callback)
        # self.misty.unregister_all_events()
        while IMU.is_active:
            pass
        return yaw_value

    def get_nearest_obstacle(self, keep_alive = False):
        tof = self.misty.register_event("TimeOfFlight", Events.TimeOfFlight, condition=None, debounce=0, keep_alive=keep_alive, callback_function=tof_call)
        
        while tof.is_active:
            pass

        return obstacle, location

    def rotate(self, goal='person'):
        self.misty.move_head(0, 0, 0)
        not_once = True
        ang = 0
        count = 0

        while True:
            try:
                self.take_image()
                rect_vertices, goal_detected = self.yolo_object_detection('/home/rahul/Desktop/misty_github/Examples/test.jpg', goal)
                x, y, w, h = rect_vertices
                print("Goal detected is ", goal_detected)
                if goal_detected:
                    if not_once:
                        heading = calculate_absolute_heading(self.get_current_yaw(), ang)
                        self.misty.drive_arc(heading = heading, radius=.01, timeMs=1000, reverse = False)
                        self.misty.move_head(0, 0, 0)
                    # angle = 0 
                    if(x+w/2 >= 120) and (x+w/2 <= 200):
                        print("Person at frame centre")
                        break
                    elif(x+w/2 < 120):
                        print("Person Left of frame centre")
                        # angle = angle + 20
                        count = 10
                        heading = calculate_absolute_heading(self.get_current_yaw(), 20)
                        self.misty.drive_arc(heading = heading, radius=.01, timeMs=1000, reverse = False)
                        self.misty.move_head(0, 0, 0)
                        not_once = False
                        time.sleep(5)
                    else:
                        print("Person Right of frame centre")
                        # angle = angle - 20
                        count = 10
                        heading = calculate_absolute_heading(self.get_current_yaw(), -20)
                        self.misty.drive_arc(heading = heading, radius=.01, timeMs=1000, reverse = False)
                        self.misty.move_head(0, 0, 0)
                        not_once = False
                        time.sleep(5)
                else:
                    pass
            
            except UnboundLocalError as e:
                if count >= 6:
                    count = 0

                if count==0:
                    self.misty.move_head(0, 0, 0)
                    # ang = 0
                    count+=1
                    pass
                elif count==1:
                    ang = 30
                    self.misty.move_head(0, 0, ang)
                    count+=1
                    time.sleep(2)
                elif count==2:
                    ang = -30
                    self.misty.move_head(0, 0, ang)
                    count+=1            
                    time.sleep(2)
                elif count==3:
                    ang = 60
                    self.misty.move_head(0, 0, ang)
                    count+=1
                    time.sleep(2)
                elif count==4:
                    ang = -60
                    self.misty.move_head(0, 0, ang)
                    count+=1
                    time.sleep(2)
                elif count==5:
                    ang = 90
                    self.misty.move_head(0, 0, ang)
                    count+=1
                    time.sleep(2)
                elif count==6:
                    ang = -90
                    self.misty.move_head(0, 0, ang)
                    count+=1
                    time.sleep(2)
                else:
                    count = 0 
                    self.misty.move_head(0, 0, 0)
                    self.drive_misty(distance=0.1, theta=0, radius=0, reverse = True)
                    time.sleep(5)
        
        print("Executed rotate---------------------now rotating misty")

        time.sleep(5)
        return None


    def drive_misty(self, distance=0, theta=0, radius=0, reverse = False):
        global_yaw_value = self.get_current_yaw()
        new_heading = calculate_absolute_heading(global_yaw_value, theta)
        self.misty.drive_heading(heading=new_heading, distance=distance, timeMs=500, reverse=reverse)


    def navigation(self, goal='person'):
        print("\n\n-------------START DRIVING-------------\n\n")
        self.misty.move_head(0, 0, 0)
        while True:
            distance, rotate = self.get_distance_image(goal)
            print(f"Distance to object is {distance} metres")
            if rotate == False:
                if distance!=np.nan and distance >= 0.70: 
                    print(f"Distance to object is {distance} metres")
                    self.misty.move_head(0, 0, 0)
                    self.drive_misty(distance=distance-0.1, theta=0, radius=0)
                    time.sleep(10)
            elif rotate == True:
                print("No person detected")
                self.rotate(goal)
                distance, _ = self.get_distance_image(goal)
                print(f"Distance to object is {distance} metres")
                self.drive_misty(distance=distance-0.1, theta=0, radius=0)
                time.sleep(10)    

            else:
                pass
            print("Triggering sensor")
            start = time.time()
            while True:
                end = time.time()
                obs, loc = self.get_nearest_obstacle(keep_alive=False)
                print(f"Elapsed time is {end-start}")
                if end-start <= 5:
                    flag = True
                    if obs > 0.1 and obs != np.inf:
                        print(f"Distance to object is {obs} metres")
                        self.drive_misty(distance=obs-0.15, theta=0, radius=0)
                    else:
                        pass
                    break
                
            if flag:
                break
        
        self.misty.unregister_all_events()
        self.navigation()
        print("\n\n------------STOP DRIVING------------\n\n")
        


if __name__ == "__main__":

    ip = '10.134.109.106'
    run = ObjectDetection(ip)
    goal = input("Press enter to track person or mention object to track: ")
    if goal == '':
        goal = 'person'
    else:
        pass
    run.navigation(goal)