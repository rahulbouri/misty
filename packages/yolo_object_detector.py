from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import os
import numpy as np
import cv2 as cv
import math
import base64
from transformers import pipeline


class YoLo:

    def __init__(self, ip):
        self.ip = ip
        self.misty = Robot(self.ip)
        self.checkpoint = "vinvino02/glpn-nyu"
        self.depth_estimator = pipeline("depth-estimation", model = self.checkpoint)
        self.labels_path = '/home/rahul/Desktop/misty_github/yolo-coco/coco.names'
        self.weights_path = '/home/rahul/Desktop/misty_github/yolo-coco/yolov3.weights'
        self.config_path = '/home/rahul/Desktop/misty_github/yolo-coco/yolov3.cfg'
        self.img_file_path = '/home/rahul/Desktop/misty_github/Examples/test.jpg'
        self.yolo_img_path = '/home/rahul/Desktop/misty_github/Examples/test_yolo.jpg'

    def take_image(self):
        self.misty.take_picture(base64=True, fileName="test", width=320, height=240, displayOnScreen=False, overwriteExisting=True)
        img_b64 = self.misty.get_image(fileName="test.jpg", base64=True).json()['result']['base64']
        image_data = base64.b64decode(img_b64)
        directory, filename = os.path.split(self.img_file_path)
        with open(os.path.join(directory, filename), "wb") as image_file:
            image_file.write(image_data)

    def detect(self, goal='person'):
        self.take_image()
        
        labels = open(self.labels_path).read().split("\n")

        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
        net = cv.dnn.readNetFromDarknet(self.config_path, self.weights_path)

        image = cv.imread(self.img_file_path)
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
                    rect_vertices = x, y, w , h
                    goal_detected = True
                    break
                elif math.sqrt(w**2 + h**2) > biggest_box:
                    biggest_box = math.sqrt(w**2 + h**2)
                    rect_vertices = x, y, w, h
                else:
                    rect_vertices = None

        # cv.imshow("Image", image)
        # cv.waitKey(0) 
        cv.imwrite(self.yolo_img_path, image)       
        
        return rect_vertices, goal_detected
           