import cv2
import mediapipe as mp
import time
import numpy as np

cap =cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
previous_time = 0
current_time = 0

while (True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = img

    frame = np.zeros(shape=img.shape, dtype=np.uint8)  # shape = [H, W]
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)

    cv2.imshow("IMG", img)
    cv2.imshow('hand',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()