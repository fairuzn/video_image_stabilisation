#%%
import cv2

input_video_path = 'Simple1.mp4'

cap = cv2.VideoCapture(input_video_path)
frame_list = []
while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame)
    frame_list.append(frame)
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()
# %%
