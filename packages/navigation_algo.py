from misty_drive_controller import Driver
from object_camera_align import Misty_Object_Aligner
from distance_calculator import Distance
from yolo_object_detector import YoLo

import time
import numpy as np


def navigation(ip, goal):
    
    if goal == '':
        goal = 'person'
    else:
        pass
    
    yolo = YoLo(ip)
    aligner = Misty_Object_Aligner(ip, goal)
    sense = Distance(ip, goal)
    driver = Driver(ip)
    print("\n\n-------------START DRIVING-------------\n\n")
    driver.centre_head()

    while True:
        distance, rotate = sense.distance_or_rotate()
        
        if distance == np.nan:
            time.sleep(5)
            distance, rotate = sense.distance_or_rotate()
            print(f"Distance to object is {distance} metres")
        
        else:
            pass

        if rotate == False:
            if distance!=np.nan and distance >= 0.70: 
                print(f"Distance to object is {distance} metres")
                driver.centre_head()
                driver.drive_misty(distance=distance-0.1, theta=0, radius=0)
                time.sleep(10)

        elif rotate == True:
            print("No person detected")
            aligner.rotate()
            distance, _ = sense.distance_or_rotate()
            print(f"Distance to object is {distance} metres")
            driver.drive_misty(distance=distance-0.1, theta=0, radius=0)
            time.sleep(10)

        else:
            pass

        print("Navigate using ToF sensor")

        start = time.time()
        while True:
            end = time.time()
            obs, loc = driver.get_nearest_obstacle(keep_alive=False)
            if end-start <= 5:
                flag = True
                if obs > 0.1 and obs != np.inf:
                    driver.drive_misty(distance=obs-0.15, theta=0, radius=0)
                else:
                    pass
                break
            
        if flag:
            break

        driver.unregister_all_events()
        navigation(goal)


ip = '10.134.109.106'
goal = input("Enter the object you want to navigate to: ")
navigation(ip, goal)