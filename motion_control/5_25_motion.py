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

            scaling_factor = 0.7 # 마우스 커서 이동값 조절 계수
            

            # idx 1 : 제스처가 'move'일 때 'IndexFingerTip' 위치를 가져와서 마우스 커서를 움직인다.
            # 제스처 'move' : 손가락 총모양.
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

                    # 상대적 좌표 이동이므로 현재 위치와 이전 위치의 차이를 계산한다.
                    diff_x = (screen_x - prev_screen_x) * scaling_factor
                    diff_y = (screen_y - prev_screen_y) * scaling_factor

                    # 위치 차이 만큼 마우스 커서 이동
                    pyautogui.move(diff_x, diff_y)

                    # 다음 루프 반복을 위해 현재 위치를 저장한다.
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
            

            # idx 2 : 제스처가 'left_click'일 때 마우스 왼쪽 버튼을 1번 누른다.
            # 제스처 'left_click' : 손가락이 총 모양일 때 엄지를 접기.
            if idx == 2:
                
                pyautogui.mouseDown(button="left")
                pyautogui.mouseUp(button="left")  
                time.sleep(2)
                
            # idx 3 : 제스처가 'right_click'일 때 마우스 오른쪽 버튼을 1번 누른다.
            # 제스처 'right_click' : 손가락을 딱 접어서 숫자 '2'만들기.
            if idx == 3:
                    
                pyautogui.mouseDown(button="right")
                pyautogui.mouseUp(button="right")
                time.sleep(2)
            
            # idx 4 : 제스처가 'scroll_up'일 때 스크롤을 100만큼 올린다.
            # 제스처 'scroll_up' : 손가락을 딱 접어서 숫자 '2'를 만든 상태로 위로 향한다.
            if idx == 4:
                    
                pyautogui.scroll(100)
            
            # idx 5 : 제스처가 'scroll_down'일 때 스크롤을 100만큼 내린다.
            # 제스처 'scroll_down' : 손가락을 딱 접어서 숫자 '2'를 만든 상태로 아래로 향한다.    
            if idx == 5:
                
                pyautogui.scroll(-100)
            
            # idx 6 : 제스처가 'left'일 때 키보드 왼쪽 화살표 버튼을 1회 누른다.
            # 제스처 'left' : 주먹을 쥔 상태에서 엄지 손가락만 곧게 편다.   
            if idx == 6:
                
                pyautogui.press('left', presses=1)
                time.sleep(0.5)
            
            # idx 7 : 제스처가 'right'일 때 키보드 오른쪽 화살표 버튼을 1회 누른다.
            # 제스처 'right' : 주먹을 쥔 상태에서 새끼 손가락만 곧게 편다.
            if idx == 7:
                
                pyautogui.press('right', presses=1)
                time.sleep(0.5)
            

            # idx 8 : 제스처가 'double_click'일 때 마우스 왼쪽 버튼 더블 클릭을 1회 한다.
            # 제스처 'double_click' : 손으로 'V'사인을 그린다.
            if idx == 8: 
                
                pyautogui.doubleClick()
                time.sleep(1)
                
            # idx 9 : 제스처가 'wheel_click'일 때 마우스 휠 클릭 버튼을 1번 누른다.
            # 제스처 'wheel_click' : 주먹을 쥔다. 
            if idx == 9:
                
                pyautogui.mouseDown(button="middle")
                pyautogui.mouseUp(button="middle")
                time.sleep(1)
            
            # idx 10 : 제스처가 'size_up'일 때 화면을 확대한다.
            # 제스처 'size_up' : 네 손가락을 딱 붙인 채로 엄지와 네 손가락의 간격을 벌린다.
            if idx == 10:
                    
                pyautogui.hotkey('ctrl', '+')
                time.sleep(0.5)
            
            # idx 12 : 제스처가 'size_down'일 때 화면을 축소한다.
            # 제스처 'size_down' : 네 손가락을 딱 붙인 채로 엄지와 네 손가락의 간격을 좁힌다.  
            if idx == 12:
                    
                pyautogui.hotkey('ctrl', '-')
                time.sleep(0.5)
            
            cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    window_name = 'HGRSCS by FourFingers'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 크기가 자유로운 웹캠 윈도우 생성
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # 창을 항상 상단에 표시
    cv2.imshow(window_name, img)

    if cv2.waitKey(1) == ord('q'):
        break