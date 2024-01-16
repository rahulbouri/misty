from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import pandas as pd

obs_list = []
loc_list = []
haz_list = []
sensor_list = []

global obstacle, location, hazard, status, sensor
def tof_call(data):
    global obstacle, location, hazard, status, sensor
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
        status = message.get('status')
        if message.get('status')==0:
            obstacle = message.get('distanceInMeters')
            hazard = message.get('inHazard')
            sensor = message.get('type')
        
            if message.get('sensorID') in ['toffr', 'toffl', 'toffc', 'tofdfr', 'tofdfl']:
                location = 'front'
            else:
                location = 'back'
            return obstacle, location, hazard
    
        else:
            obstacle = 1
            location = 'front'
            hazard = False
            status = None
    else:
        print("Data format is not as expected.")

def get_nearest_obstacle(ip, keep_alive = False):
    misty = Robot(ip)
    tof = misty.register_event("TimeOfFlight", Events.TimeOfFlight, condition=None, debounce=0, keep_alive=keep_alive, callback_function=tof_call)
    
    while tof.is_active:
        pass
    return obstacle, location, hazard

def main():
    global obstacle, location, hazard, status, obs_list, loc_list, haz_list, sensor_list
    hazard = False
    ip = '192.168.1.101'
    count = 0 
    while (len(obs_list) < 50):
        obstacle, location, hazard = get_nearest_obstacle(ip, keep_alive = False)
        if status == 0:
            count += 1
            obs_list.append(obstacle)
            loc_list.append(location)
            haz_list.append(hazard)
            sensor_list.append(sensor)
            print("Data points collected", count)
        

    print("------Converting to Dataframe-------")

    data = {
    'Obstacle': obs_list,
    'Location of Obstacle': loc_list,
    'Hazard(Boolean)': haz_list,
    'Sensor Type': sensor_list
}
    df = pd.DataFrame(data)

    df.to_csv('/home/rahul/Desktop/misty_github/Examples/obstacle_data.csv', index=False)



main()


























# def hazard_callback(data):
#     print("Data")
#     if 'message' in data:
#         message = data['message']
#         print("\n-----------------\n")
#         print("Hazard Notification Event:")
#         print("\n-----------------\n")

#         # Process bumpSensorsHazardState
#         print("Bump Sensors Hazard State:")
#         for sensor in message.get('bumpSensorsHazardState'):
#             print("Sensor Name:", sensor['sensorName'])
#             print("In Hazard:", sensor['inHazard'])
#             print("\n")

#         # Process driveStopped
#         print("Drive Stopped Sensors:")
#         for sensor in message.get('driveStopped'):
#             print("Sensor Name:", sensor['sensorName'])
#             print("In Hazard:", sensor['inHazard'])
#             print("\n")

#         # Process timeOfFlightSensorsHazardState
#         print("Time of Flight Sensors Hazard State:")
#         for sensor in message.get('timeOfFlightSensorsHazardState'):
#             print("Sensor Name:", sensor['sensorName'])
#             print("In Hazard:", sensor['inHazard'])
#             print("\n")

#         print("\n-----------------\n")
#     else:
#         print("Invalid data format. The 'message' key is missing.")

# def hazard_callback(data):
#     print("Data")
#     if 'message' in data:
#         message = data['message']
#         hazard = False
#         for sensor in message.get('bumpSensorsHazardState'):
#             hazard = sensor['inHazard']
            
#             print(f"Hazard is {hazard}")

#             if hazard == True:
#                 break

#         for sensor in message.get('driveStopped'):
#             hazard = sensor['inHazard']

#             print(f"Hazard is {hazard}")

#             if hazard == True:
#                 break

#         for sensor in message.get('timeOfFlightSensorsHazardState'):
#             hazard = sensor['inHazard']

#             print(f"Hazard is {hazard}")
            
#             if hazard == True:
#                 break
#     else:
#         print("Invalid data format. The 'message' key is missing.")


# def hazard_notif(ip, keep_alive = False):
#     misty = Robot(ip)
#     print(misty.get_battery_level().json())
#     notif = misty.register_event("HazardNotification", Events.HazardNotification, condition=None, debounce=0, keep_alive=keep_alive, callback_function=hazard_callback)
#     while notif.is_active:
#         pass

# start = time.time()
# hazard_notif('192.168.1.101', keep_alive = False)
# end = time.time()
# print(f"Time taken is {end-start}")