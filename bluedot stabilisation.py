import numpy as np
import cv2

cap = cv2.VideoCapture('Simple1.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame)
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        break
    
cap.release()