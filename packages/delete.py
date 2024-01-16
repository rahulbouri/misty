import cv2
import os
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def fallen(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    if (a[1]>b[1]) and abs(a[1]-c[1]) < 10 and abs(b[1]-c[1]) < 10:
        return True
    else:
        return False
    
def determine_fall(landmarks):
    try:
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # print("Nose: ", nose)
        # print("Left Hip: ", left_hip)
        # print("Right Hip: ", right_hip)
        # print("Left Knee: ", left_knee)
        # print("Right Knee: ", right_knee)
        # print("Left Ankle: ", left_ankle)
        # print("Right Ankle: ", right_ankle)

        if fallen(nose, left_hip, left_ankle):
            return True
        elif fallen(nose, right_hip, right_ankle):
            return True
        elif left_hip[1] < right_knee[1] or right_hip[1] < left_knee[1]:
            return True
        else:
            pass

    except:
        return False

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle 

def determine_sit(landmarks):
    try:
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        if calculate_angle(left_shoulder, left_hip, left_knee)<120 and calculate_angle(left_shoulder, left_hip, left_knee)>60:
            return True
        elif calculate_angle(right_shoulder, right_hip, right_knee)<120 and calculate_angle(right_shoulder, right_hip, right_knee)>60:
            return True
        else:
            return False
        
    except:
        return False


def process_image(image_path):
    fallen = False
    image = cv2.imread(image_path)
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

    cv2.imshow('Mediapipe Feed', image)

    if determine_fall(landmarks):
        fallen = True
    
    print("Fallen Person Detected: ", fallen)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return fallen


def process_sit(image_path):
    sit = False
    image = cv2.imread(image_path)
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
    cv2.imshow('Mediapipe Feed', image)

    if determine_sit(landmarks):
        print("Sitting Person Detected")
        sit = True

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return sit


if __name__ == '__main__':
    image_path = '/home/rahul/Desktop/human_activity/lying/14.jpeg'
    fallen = process_image(image_path)