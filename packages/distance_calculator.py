from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

from yolo_object_detector import YoLo

import os
import base64
import requests
import numpy as np


class Distance:

    def __init__(self, ip, goal):
        self.ip = ip
        self.goal = goal
        self.misty = Robot(self.ip)
        self.img_file_path = '/home/rahul/Desktop/misty_github/Examples/test.jpg'
        self.yolo = YoLo(self.ip)


    def take_image(self):
        self.misty.take_picture(base64=True, fileName="test", width=320, height=240, displayOnScreen=False, overwriteExisting=True)
        img_b64 = self.misty.get_image(fileName="test.jpg", base64=True).json()['result']['base64']
        image_data = base64.b64decode(img_b64)
        directory, filename = os.path.split(self.img_file_path)
        with open(os.path.join(directory, filename), "wb") as image_file:
            image_file.write(image_data)

    def distance_or_rotate(self):
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
                rect_vertices, _ = self.yolo.detect(self.goal)
                x, y, w, h = rect_vertices
                roi = dist_image[x:x + w, y:y + h]
                valid_values = roi[~np.isnan(roi)]

                if len(valid_values) > 0:
                    distance = np.min(valid_values)
                    # break
                else:
                    distance = np.nan
                    rotate = True
                
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