#importing misty libraries which gives functionality to connect and control the robot
from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import os
import base64
import torch
import time 

ip='10.134.109.106' #ip address of the robot, you can check it out from the misty app
misty = Robot(ip)   #creating a misty object to control the robot

def main():
    misty.enable_av_streaming_service()
    input("Press Enter to start the program ")
    # x = float(input("How long should stream last (in seconds):  "))
    misty.start_av_streaming(url="rtspd:1935", width=640, height=480)
    input("Press Enter to stop the program ")
    misty.stop_av_streaming()



if __name__ == '__main__':
    misty.stop_av_streaming()
    main()
    misty.disable_av_streaming_service()
    print("Finished")