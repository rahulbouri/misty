#importing misty libraries which gives functionality to connect and control the robot
from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import base64
import io

ip='192.168.1.103' #ip address of the robot, you can check it out from the misty app
misty = Robot(ip)

def main():

    while True:
        key = input("Press enter to record video or type \'QUIT\' ")
        key = key.lower()

        misty.enable_camera_service() #camera service must be enabled to start recording video and also download video

        if key!='quit':
            x = int(input("How long do you want to record video in seconds, max duration is 3 minutes i.e 180 seconds? "))
            print("Recording Started")
            misty.start_recording_video(fileName='test', mute=False, duration=x, width=1920, height=1080) #specifying recording settings
            print("Recording Done")
        else:
            break
        
    print('Downloading recorded video')
    b64_video=misty.get_video_recording(name='test', base64=True).json()['result']['base64']
    mp4 = base64.b64decode(b64_video) #decoding base64 string to mp4 format
    with open('temp.mp4', "wb") as output_file:
        output_file.write(mp4)
    
    misty.disable_camera_service() #turn camera service off once requirement over to conserve batter

main()