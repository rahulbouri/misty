from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import sys

sys.path.insert(0, "/home/rahul/Desktop/misty_github/")

from DistDepth.networks.resnet_encoder import ResnetEncoder
from DistDepth.networks.depth_decoder import DepthDecoder
from DistDepth.utils import output_to_depth

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

        if message.get('status')==0:
            left = message.get('distanceInMeters') if message.get('sensorId') == 'tofdfl' else 1
            right = message.get('distanceInMeters') if message.get('sensorId') == 'tofdfr' else 1
            centre = message.get('distanceInMeters') if message.get('sensorId') == 'toffc' else 1
            obstacle = min(left, right, centre)

            # if message.get('sensorID') in ['toffr', 'toffl', 'toffc', 'tofdfr', 'tofdfl']:
            #     obstacle = message.get('distanceInMeters')
            #     hazard = message.get('inHazard')
            #     location = 'front'
            # else:
            #     location = 'back'
            # else:
            #     location = 'back'
        
        else:
            obstacle = 1
            location = 'front'
            hazard = False
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
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cudnn.enabled = True 
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        with torch.no_grad():
            print("Loading Pretrained Model")
            self.encoder = ResnetEncoder(152, False)
            loaded_dict_enc = torch.load(
                "/home/rahul/Desktop/DistDepth/ckpts/encoder.pth",
                map_location=self.device,
            )
            filtered_dict_enc = {
                k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()
            }
            self.encoder.load_state_dict(filtered_dict_enc)
            self.encoder.to(self.device)
            self.encoder.eval()

            self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

            loaded_dict = torch.load(
                "/home/rahul/Desktop/DistDepth/ckpts/depth.pth",
                map_location=self.device,
            )
            self.depth_decoder.load_state_dict(loaded_dict)

            self.depth_decoder.to(self.device)
            self.depth_decoder.eval()

        
        self.misty = Robot(self.ip)
        self.misty.set_default_volume(volume=30)
        self.yaw = None

    def image_detection(self, img_file):

        # self.misty.speak("Calculating Ditance", speechRate=0.2)

        raw_img = np.transpose(cv2.imread(img_file, -1)[:, :, :3], (2, 0, 1))
        input_image = torch.from_numpy(raw_img).float().to(self.device)
        input_image = (input_image / 255.0).unsqueeze(0)

        input_image = torch.nn.functional.interpolate(
                input_image, (256, 256), mode="bilinear", align_corners=False
            )

        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)

        out = outputs[("out", 0)]

        out_resized = torch.nn.functional.interpolate(
                out, (512, 512), mode="bilinear", align_corners=False
            )
        # convert disparity to depth
        depth = output_to_depth(out_resized, 0.1, 10)
        metric_depth = depth.cpu().detach().numpy().squeeze()

        distance = metric_depth.min()

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

    def get_nearest_obstacle(self, keep_alive = False):
        tof = self.misty.register_event("TimeOfFlight", Events.TimeOfFlight, condition=None, debounce=0, keep_alive=keep_alive, callback_function=tof_call)
        
        while tof.is_active:
            pass
        
        print(f"Obstacle distance is {obstacle} metres")

        if obstacle < 0.030:
            self.drive_misty(distance=0.1, theta=180)
        return obstacle


    def drive_misty(self, distance=0, theta=0, radius=0):
        global_yaw_value = self.get_current_yaw()
        new_heading = calculate_absolute_heading(global_yaw_value, theta)
        self.misty.drive_heading(heading=new_heading, distance=distance, timeMs=500, reverse=False)


    def navigation(self):

        self.take_image()
        distance = self.image_detection(img_file='/home/rahul/Desktop/misty_github/Examples/test.jpg')
        print(f"Object distance is {distance} metres")
        self.get_nearest_obstacle()
        


        print("\n\n-------------START DRIVING-------------\n\n")

        while (distance>=0.5):
            print(f"Distance is {distance} metres")
            self.get_nearest_obstacle()
            self.drive_misty(distance=float(distance), theta=0, radius=0)
            self.get_nearest_obstacle()
            self.take_image()
            distance = self.image_detection(img_file='/home/rahul/Desktop/misty_github/Examples/test.jpg')

        print("\n\n------------STOP DRIVING------------\n\n")
        print("Obstacle detected at ", obstacle)
        self.misty.unregister_all_events()


if __name__ == "__main__":

    ip = '192.168.1.101' #ip address of the robot, you can check it out from the misty app
    run = ObjectDetection(ip)
    run.navigation()

