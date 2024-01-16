from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

from yolo_object_detector import YoLo

import os
import time
import base64

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
    

class Misty_Object_Aligner:

    def __init__(self, ip, goal):
        self.ip = ip
        self.goal = goal
        self.img_file_path = '/home/rahul/Desktop/misty_github/Examples/test.jpg'
        self.misty = Robot(self.ip)
        self.yolo = YoLo(self.ip)

    def calculate_absolute_heading(self, current_heading, heading_difference):
        # Calculate the absolute heading using the 0 to 360 degrees representation.
        absolute_heading = (int(current_heading) + int(heading_difference)) % 360.0
        # Convert to the -180 to 180 degrees representation if necessary.
        if absolute_heading > 180:
            absolute_heading -= 360
        return absolute_heading

    def get_current_yaw(self):
        IMU = self.misty.register_event("IMU", Events.IMU, condition=None, debounce=0, keep_alive=False, callback_function=imu_callback)
        while IMU.is_active:
            pass
        return yaw_value

    def take_image(self):
        self.misty.take_picture(base64=True, fileName="test", width=320, height=240, displayOnScreen=False, overwriteExisting=True)
        img_b64 = self.misty.get_image(fileName="test.jpg", base64=True).json()['result']['base64']
        image_data = base64.b64decode(img_b64)
        directory, filename = os.path.split(self.img_file_path)
        with open(os.path.join(directory, filename), "wb") as image_file:
            image_file.write(image_data)
    
    def rotate(self):
        self.misty.move_head(0, 0, 0)
        print("------------Misty is now rotating------------")
        count = 0

        while True:

            try:
                count+=1
                print(f"Count is {count}")
                if count<=5:
                    angle=20
                elif count == 6:
                    angle = 0
                elif count>6 and count<=10:
                    angle = -20
                else:
                    print("No "+self.goal+" detected in FoV of Misty")
                    count = 0 
                    heading = self.calculate_absolute_heading(self.get_current_yaw(), +100)
                    self.misty.drive_arc(heading = heading, radius=.01, timeMs=1000, reverse = False)
                    reverse = input("How much do you want to reverse?: ")
                    time.sleep(5)
                    self.drive_misty(distance=reverse, theta=0, radius=0, reverse = True)
                    time.sleep(5)

                if count !=6:
                    heading = self.calculate_absolute_heading(self.get_current_yaw(), angle)
                    self.misty.drive_arc(heading = heading, radius=.01, timeMs=1000, reverse = False)
                else:
                    heading = self.calculate_absolute_heading(self.get_current_yaw(), -100)
                    self.misty.drive_arc(heading = heading, radius=.01, timeMs=1000, reverse = False)

                time.sleep(5)
                self.take_image()
                rect_vertices, _ = self.yolo.detect(self.goal)
                x, y, w, h = rect_vertices

                if (x+w/2)>=100 and (x+w/2)<=220:
                    print("Person detected")
                    break
                else:
                    pass
            
            except UnboundLocalError as e:
                pass

        print("------------Misty is now facing the person------------")
        self.misty.move_head(0, 0, 0)

        return None
    
    def patrolling_rotation(self):
        self.misty.move_head(0, 0, 0)
        heading = self.calculate_absolute_heading(self.get_current_yaw(), +15)
        self.misty.drive_arc(heading = heading, radius=0.01, timeMs=1000, reverse = False)
        time.sleep(2)
        return None
        
