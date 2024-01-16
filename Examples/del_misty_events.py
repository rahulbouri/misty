from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import base64
import os

misty = Robot('10.134.109.106')

misty.take_picture(base64=True, fileName="test", width=320, height=240, displayOnScreen=False, overwriteExisting=True)
img_b64 = misty.get_image(fileName="test.jpg", base64=True).json()['result']['base64']
image_data = base64.b64decode(img_b64)
directory = '/home/rahul/Desktop/misty_github/Examples'
filename = 'delete.jpg'
with open(os.path.join(directory, filename), "wb") as image_file:
    image_file.write(image_data)