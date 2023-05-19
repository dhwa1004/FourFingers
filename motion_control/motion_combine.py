import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from PIL import ImageFont, ImageDraw, Image

max_num_hands = 1
gesture = {
    1:'move', 2:'left_click', 3:'right_click',
    4:'scroll_up', 5:'scroll_down', 6:'left_key', 7:'right_key',
    8: 'double_click', 9:'wheel_click', 10:'size_up', 11:'drag', 12:'size_down'
}

mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

pre = ''
offset = 150

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition model
file = np.genfromtxt('gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

prev_screen_x = 0
prev_screen_y = 0

while cap.isOpened():
    ret, img = cap.read()
    
    h,w,c = img.shape
    
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img.flags.writeable = False
    results = hands.process(img)
    
    result = hands.process(img)
    
    img.flags.writeable = True
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    image_height, image_width, _ = img.shape
    
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            
            joint = np.zeros((21, 3))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            scaling_factor = 1.0
            

            # when gesture is 'move', get the 'IndexFingerTip' position and move the mouse cursor
            if idx == 1:
                image_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*image_width
                image_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*image_height

                if image_x > offset and image_x < w - offset and image_y > offset and image_y < h - offset:
                    image_x = image_x - offset
                    image_y = image_y - offset
                    new_image_height = image_height - offset*2
                    new_image_width = image_width - offset*2

                    screen_y = image_y*screen_height/new_image_height
                    screen_x = image_x*screen_width/new_image_width

                    # For relative movement, calculate the difference between the current and previous positions
                    diff_x = (screen_x - prev_screen_x) * scaling_factor
                    diff_y = (screen_y - prev_screen_y) * scaling_factor

                    # Move the mouse cursor by the difference in position
                    pyautogui.move(diff_x, diff_y)

                    # Save the current position for the next loop iteration
                    prev_screen_x = screen_x
                    prev_screen_y = screen_y

                    pre = 'move'
                
    if result.multi_hand_landmarks is not None:
        
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
                    
    if result.multi_hand_landmarks is not None:
        
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Inference gesture
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])
            
            if idx == 2:
                
                pyautogui.mouseDown(button="left")
                pyautogui.mouseUp(button="left")  
                time.sleep(1.0)
                
            if idx == 3:
                    
                pyautogui.mouseDown(button="right")
                pyautogui.mouseUp(button="right")
                time.sleep(1.0)
                
            if idx == 4:
                    
                pyautogui.scroll(100)
                time.sleep(0.5)
                
            if idx == 5:
                
                pyautogui.scroll(-100)
                time.sleep(0.5)
                
            if idx == 6:
                
                pyautogui.press('left', presses=1)
                time.sleep(1.0)
                
            if idx == 7:
                
                pyautogui.press('right', presses=1)
                time.sleep(1.0)
                
            if idx == 8: #더블클릭=브이
                
                pyautogui.doubleClick() #더블클릭
                time.sleep(1.0)
                
                
            if idx == 9: 
                
                pyautogui.mouseDown(button="middle")
                pyautogui.mouseUp(button="middle")
                time.sleep(1.0)
                
            if idx == 10:
                    
                pyautogui.hotkey('ctrl', '+')
                time.sleep(0.7)
                
            if idx == 12:
                    
                pyautogui.hotkey('ctrl', '-')
                time.sleep(0.7)
            
            cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Filter', img)
    if cv2.waitKey(1) == ord('q'):
        break