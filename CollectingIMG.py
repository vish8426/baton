import os
from os import path
import time
import uuid
import cv2
import mediapipe as mp
import numpy as np

def yes_or_no(question):
        reply = str(input(question)).lower().strip()
        if reply[0] == 'y':
            return True
        elif reply[0] == 'n':
            return False
        else:
            return yes_or_no('please enter y/n')


IMG_PATH = 'Tensorflow/workspace/images/collectedimages'

Name = input("Enter your name: ")
Name = str(Name + '_')

labels = [Name + 'forward',
          Name + 'backward',
          Name + 'faster',
          Name + 'slower',
          Name + 'left',
          Name + 'right',
          Name + 'stop']

number_imgs = 3
imgnum = 0
origin_dir = os.getcwd()
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

for label in labels:
    os.chdir(origin_dir)
    imgnum = 0
    directory = 'Tensorflow\workspace\images\collectedimages\_' + format(label)
    if os.path.isdir(directory):
        if not yes_or_no('file name existed, do you want to overwrite the file, all existing files will be loss (y/n): '):
            break
        else:
            os.removedirs(directory)
    os.mkdir(directory)
    os.chdir(directory)
    print('Collecting image for {}'.format(label))
    print('to capture press C, to Quit press Q')
    while imgnum < number_imgs:
        ret, img = cap.read()
        if ret == True:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = np.zeros(shape=img.shape, dtype=np.uint8)  # shape = [H, W]
            results = hands.process(imgRGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            cv2.imshow('Frame', frame)
            cv2.imshow('IMG', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                img_name = label + '{}.jpg'.format(str(uuid.uuid1()))
                cv2.imwrite(img_name, frame)
                print('captured ' + img_name)
                imgnum += 1
                print('image ' + str(imgnum) + ' captured')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

