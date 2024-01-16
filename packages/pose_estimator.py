from mistyPy.Robot import Robot
from mistyPy.Events import Events
from mistyPy.RobotCommands import RobotCommands

import time
import base64
import os

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class PoseEstimator:
    def __init__(self, ip):
        self.ip = ip
        self.misty = Robot(self.ip)
        self.mp_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp_drawing
        self.pose = None
        self.pose_landmarks = None
        self.img_file_path = '/home/rahul/Desktop/misty_github/packages/test.jpg'
    
    def take_image(self):
        self.misty.take_picture(base64=True, fileName="test", width=320, height=240, displayOnScreen=False, overwriteExisting=True)
        img_b64 = self.misty.get_image(fileName="test.jpg", base64=True).json()['result']['base64']
        image_data = base64.b64decode(img_b64)
        directory, filename = os.path.split(self.img_file_path)
        with open(os.path.join(directory, filename), "wb") as image_file:
            image_file.write(image_data)

    # def start_misty_av_stream(self):
    #     self.misty.stop_av_streaming()
    #     self.misty.enable_av_streaming_service()
    #     print("AV Streaming Service Started")
    #     duration = input("Enter duration of streaming in seconds: ")
    #     self.misty.start_av_streaming(url="rtspd:1935", width=640, height=480)
    #     time.sleep(duration)
    #     self.misty.stop_av_streaming()
    #     self.misty.disable_av_streaming_service()
    
    # def stop_misty_av_stream(self):
    #     self.misty.stop_av_streaming()
    #     self.misty.disable_av_streaming_service()

    # def process_stream(self):
    #     fallen = False
    #     self.start_misty_av_stream()
    #     stream_url = 'rtsp://10.134.109.106:1935'
    #     cap = cv2.VideoCapture(stream_url)
    #     count = 1
    #     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            
    #         while cap.isOpened():
    #             ret, frame = cap.read()

    #             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             image.flags.writeable = False

    #             results = pose.process(image)

    #             image.flags.writeable = True
    #             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #             try:
    #                 landmarks = results.pose_landmarks.landmark
    #             except:
    #                 pass
                
    #             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
    #                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
    #                                      )               
                
    #             cv2.imshow('Mediapipe Feed', image)
    #             print("Saving image")
    #             print(type(image))
    #             cv2.imwrite('/home/rahul/Desktop/misty_github/packages/'+str(count)+'.jpg', image)
    #             print("Image saved")
    #             count+=1

    #             if self.determine_fall(landmarks):
    #                 print("Fallen Person Detected")
    #                 fallen = True
    #                 self.stop_misty_av_stream()
    #                 break

    #             if cv2.waitKey(10) & 0xFF == ord('q'):
    #                 break
                
    #         cap.release()
    #         cv2.destroyAllWindows()
        
    #     return fallen
    
    # def get_pose(self):
    #     self

    def process_sit(self):
        sit = False
        self.take_image()
        image = cv2.imread(self.img_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            landmarks = None

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # cv2.imshow('Mediapipe Feed', image)

        if self.determine_sit(landmarks):
            sit = True
            
        print("Sitting Person Detected", sit)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return sit
    
    def process_image(self):
        fallen = False
        self.take_image()
        image = cv2.imread(self.img_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            landmarks = None

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # cv2.imshow('Mediapipe Feed', image)

        if self.determine_fall(landmarks):
            fallen = True
    
        print("Fallen Person Detected: ", fallen)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return fallen

    
    def fallen(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        if (a[1]>b[1]) and abs(a[1]-c[1]) < 10 and abs(b[1]-c[1]) < 10:
            return True
        else:
            return False
    
    def determine_sit(self, landmarks):
        try:
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            if self.calculate_angle(left_shoulder, left_hip, left_knee)<120 and self.calculate_angle(left_shoulder, left_hip, left_knee)>60:
                return True
            elif self.calculate_angle(right_shoulder, right_hip, right_knee)<120 and self.calculate_angle(right_shoulder, right_hip, right_knee)>60:
                return True
            else:
                return False
        except:
            return False


    def determine_fall(self, landmarks):
        try:
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            if self.fallen(nose, left_hip, left_ankle):
                return True
            elif self.fallen(nose, right_hip, right_ankle):
                return True
            elif left_hip[1] < right_knee[1] or right_hip[1] < left_knee[1]:
                return True
            else:
                pass

        except:
            return False
        
    
    def calculate_angle(self,a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle

        return angle 