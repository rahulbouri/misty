import sys, os

from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

ip='192.168.1.101'

misty = Robot(ip)

misty.move_arm('right', 90, 100)