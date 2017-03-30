import argparse

import cv2
import numpy as np

import time

parser = argparse.ArgumentParser(description='OpenCV script to capture written content from video capture device before erasure/page turn')
parser.add_argument('--path', nargs='?', default='./store/')
args = parser.parse_args()

path = args['path']

cap = cv2.VideoCapture(0)
kernel = np.ones((3,3),np.uint8)

last100frames = []
last100area = []
num_frames = -1

background = 500
shot_number = 0

while(True):
    time.sleep(0.5)
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)
    
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.medianBlur(grayscale,5)
    
    threshold = cv2.adaptiveThreshold(grayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    erosion = cv2.erode(cv2.bitwise_not(threshold),kernel,iterations = 1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    area = cv2.countNonZero(opening)
    
    num_frames += 1
    if num_frames%1==0: #was 15 earlier
        if area >= background:
            last100frames.append(frame)
            last100area.append(area)
        elif len(last100area) > 10: #empty region, save shot with maximum area among last 100
            cv2.imwrite(path + 'shot-' + str(shot_number) + '.jpg', last100frames[np.argmax(last100area)])
            last100frames = []
            last100area = []
            shot_number += 1
            
    if len(last100area) > 100:
        last100frames = last100frames[-100:]
        last100area = last100area[-100:]
    
    #debugg
    #print area,
    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
