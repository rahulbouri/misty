from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import numpy as np

global obstacle, location, hazard


def tof_call(data):

    global obstacle, location, hazard
    if 'message' in data:
        message = data['message']

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
        
        else:
            obstacle = np.inf
            location = 'centre'
    else:
        print("Data format is not as expected.")
        
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

class Driver:

    def __init__(self, ip):
        self.ip = ip
        self.misty = Robot(self.ip)

    def get_current_yaw(self):
        IMU = self.misty.register_event("IMU", Events.IMU, condition=None, debounce=0, keep_alive=False, callback_function=imu_callback)
        while IMU.is_active:
            pass
        return yaw_value

    def centre_head(self):
        self.misty.move_head(0, 0, 0)

    def unregister_all_events(self):
        self.misty.unregister_all_events()

    def get_nearest_obstacle(self, keep_alive = False):
        tof = self.misty.register_event("TimeOfFlight", Events.TimeOfFlight, condition=None, debounce=0, keep_alive=keep_alive, callback_function=tof_call)
        
        while tof.is_active:
            pass

        return obstacle, location
    
    def calculate_absolute_heading(self, current_heading, heading_difference):
        # Calculate the absolute heading using the 0 to 360 degrees representation.
        absolute_heading = (int(current_heading) + int(heading_difference)) % 360.0
        # Convert to the -180 to 180 degrees representation if necessary.
        if absolute_heading > 180:
            absolute_heading -= 360
        return absolute_heading
    
    def drive_misty(self, distance=0, theta=0, radius=0, reverse = False):
        global_yaw_value = self.get_current_yaw()
        new_heading = self.calculate_absolute_heading(global_yaw_value, theta)
        self.misty.drive_heading(heading=new_heading, distance=distance, timeMs=500, reverse=reverse)

