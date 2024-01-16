from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

ip='192.168.1.101'

misty=Robot(ip)

response = misty.capture_speech_azure(
    overwriteExisting=True,
    silenceTimeout=10000,
    maxSpeechLength=10000,
    requireKeyPhrase=True,
    captureFile=True,
    speechRecognitionLanguage="en-US",
    azureSpeechKey="your_azure_speech_key",
    azureSpeechRegion="your_azure_speech_region"
).json()

