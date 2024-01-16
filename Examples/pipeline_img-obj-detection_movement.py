from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

from ...DistDepth.networks.resnet_encoder import ResnetEncoder
from ...DistDepth.networks.depth_decoder import DepthDecoder
from ...DistDepth.utils import output_to_depth

import base64
import time

import torch
import cv2
import random
import os,time
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle as pkl
import argparse
import threading, queue
from torch.multiprocessing import Pool, Process, set_start_method
from util import write_results, load_classes
from preprocess import letterbox_image
from darknet import Darknet
from imutils.video import WebcamVideoStream,FPS

# #  Setting up torch for gpu utilization
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

current_yaw = None

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


labels = {}
b_boxes = {}

global yaw_value


def imu_callback(data):
    global yaw_value 

    if 'message' in data:
        imu_data = data['message']
        # print("IMU Data:")
        # print("Created:", imu_data.get('created'))
        # print("Expiry:", imu_data.get('expiry'))
        # print("Pitch:", imu_data.get('pitch'))
        # print("Pitch Velocity:", imu_data.get('pitchVelocity'))
        # print("Roll:", imu_data.get('roll'))
        # print("Roll Velocity:", imu_data.get('rollVelocity'))
        # print("Sensor ID:", imu_data.get('sensorId'))
        # print("Sensor Name:", imu_data.get('sensorName'))
        # print("X Acceleration:", imu_data.get('xAcceleration'))
        # print("Y Acceleration:", imu_data.get('yAcceleration'))
        # print("Yaw:", imu_data.get('yaw'))
        # print("Yaw Velocity:", imu_data.get('yawVelocity'))
        # print("Z Acceleration:", imu_data.get('zAcceleration'))
        yaw_value = imu_data.get('yaw')

        return yaw_value
    else    :
        print("IMU data format is not as expected.")

        return None


def calculate_absolute_heading(current_heading, heading_difference):
    # Calculate the absolute heading using the 0 to 360 degrees representation.
    absolute_heading = (int(current_heading) + int(heading_difference)) % 360.0

    # Convert to the -180 to 180 degrees representation if necessary.
    if absolute_heading > 180:
        absolute_heading -= 360

    return absolute_heading



def write(bboxes, img, classes, colors):
    """
        Draws the bounding box in every frame over the objects that the model detects
    """
    class_idx = bboxes
    bboxes = bboxes[1:5]
    bboxes = bboxes.cpu().data.numpy()
    bboxes = bboxes.astype(int)
    b_boxes.update({"bbox": bboxes.tolist()})
    # bboxes = bboxes + [150,100,200,200] # personal choice you can modify this to get distance as accurate as possible
    bboxes = torch.from_numpy(bboxes)
    cls = int(class_idx[-1])
    label = "{0}".format(classes[cls])
    labels.update({"Current Object": label})
    color = random.choice(colors)

    ## Put text configuration on frame
    text_str = '%s' % (label) 
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    text_pt = (int(bboxes[0]), int(bboxes[1]) - 3)
    text_color = [255, 255, 255]

    ## Distance Meaasurement for each bounding box
    x, y, w, h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    ## item() is used to retrieve the value from the tensor
    distance = (2 * 3.14 * 180) / (w.item()+ h.item() * 360) * 1000 + 3 ### Distance measuring in Inch 
    feedback = ("{}".format(labels["Current Object"]) + " " +"is"+" at {} ".format(round(distance)) +"Inches")
    # # speak.Speak(feedback)     # If you are running this on linux based OS kindly use espeak. Using this speaking library in winodws will add unnecessary latency 
    print(feedback)

    print("{:.2f} Inches".format(distance))
    print(type(bboxes[0]))
    cv2.putText(img, str("{:.2f} Inches".format(distance)), (text_w+x.item(), y.item()), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0,255,0), font_thickness, cv2.LINE_AA)
    cv2.rectangle(img, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), color, 2)
    cv2.putText(img, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img, distance


class ObjectDetection:
    def __init__(self, ip):
        self.ip = ip
        self.cfgfile = "cfg/yolov3.cfg"
        self.weightsfile = "/home/rahul/Desktop/Object-Detection-and-Distance-Measurement/yolov3.weights"
        self.confidence = float(0.6)
        self.nms_thesh = float(0.8)
        self.num_classes = 80
        self.classes = load_classes('data/coco.names')
        self.colors = pkl.load(open("/home/rahul/Desktop/Object-Detection-and-Distance-Measurement/pallete", "rb"))
        self.model = Darknet(self.cfgfile)
        self.CUDA = torch.cuda.is_available()
        self.model.load_weights(self.weightsfile)
        self.model.net_info["height"] = 160
        self.inp_dim = int(self.model.net_info["height"])
        self.width = 640#1280
        self.height = 360#720
        print("Loading network.....")
        if self.CUDA:
            self.model.cuda()
        print("Network successfully loaded")
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32
        self.model.eval()
        self.misty = Robot(self.ip)
        self.misty.set_default_volume(volume=30)
        self.yaw = None

    def image_detection(self, img_file):

        distance = float(0)
        self.misty.speak("Tracking Person", speechRate=0.5)
        frame = cv2.imread(img_file)
        frame = cv2.resize(frame, (self.width, self.height))

        img, orig_im, dim = prep_image(frame, self.inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)
        if self.CUDA:  #### If you have a gpu properly installed then it will run on the gpu
            im_dim = im_dim.cuda()
            img = img.cuda()
        # with torch.no_grad():               #### Set the model in the evaluation mode
        output = self.model(Variable(img), self.CUDA)
        output = write_results(output, self.confidence, self.num_classes, nms=True,
                               nms_conf=self.nms_thesh)  #### Localize the objects in a frame
        output = output.type(torch.float)

        if list(output.size()) == [1, 86]:
            print(output.size())
            pass
        else:
            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(self.inp_dim)) / self.inp_dim
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]
            for boxes in output:
                frame, distance = write(boxes, frame, self.classes, self.colors)

        cv2.imshow("Object Detection Window", frame)
        cv2.waitKey(3000)

        return distance

    
    def take_image(self):
        self.misty.take_picture(base64=True, fileName="test", width=640, height=480, displayOnScreen=False, overwriteExisting=True)
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


    def drive_misty(self, distance, radius, theta):
        global_yaw_value = self.get_current_yaw()
        new_heading = calculate_absolute_heading(global_yaw_value, theta)
        # self.misty.drive_time(linearVelocity=22, angularVelocity=0,timeMs=time).json()
        # self.misty.drive_heading(heading=45, distance=distance*0.0254, timeMs=1000, reverse=False)
        self.misty.drive_arc(heading=new_heading, radius=radius, timeMs=500, reverse=False)
        time.sleep(3)
        self.misty.drive_heading(heading=new_heading, distance=distance, timeMs=500, reverse=False)
        time.sleep(1)
        self.misty.drive_arc(heading=new_heading-90, radius=0.1, timeMs=500, reverse=False)
        

def main():
    ip = '192.168.1.101' #ip address of the robot, you can check it out from the misty app
    run = ObjectDetection(ip)
    run.take_image()
    distance = run.image_detection(img_file='/home/rahul/Desktop/misty_github/Examples/test.jpg')

    while distance >=1:
        speech = "Current distance is "+ str(round(distance,1)) + " inches"
        run.misty.speak(speech, speechRate=0.1)
        run.drive_misty(distance=distance)
        time.sleep(distance*254/1000)
        run.misty.speak("Updating the distance", speechRate=0.5)
        time.sleep(3)
        run.take_image()
        distance = run.image_detection(img_file='/home/rahul/Desktop/misty_github/Examples/test.jpg')
        

if __name__ == "__main__":
    # main()
    ip = '192.168.1.101' #ip address of the robot, you can check it out from the misty app
    # run = ObjectDetection(ip)
    mia = Robot(ip)
    print(mia.get_battery_level().json())
    # run.take_image()
    # yaw = run.get_current_yaw()
    # run.drive_misty(2,0.01,-90)
    # run.drive_misty(distance=20)

