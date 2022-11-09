#%%
import cv2
import numpy as np

ix, iy, k = 200,200,1
def onMouse(event, x, y, flag, param):
	global ix,iy,k
	if event == cv2.EVENT_LBUTTONDOWN:
		ix,iy = x,y 
		k = -1

cv2.namedWindow("window")
cv2.setMouseCallback("window", onMouse)

cap = cv2.VideoCapture(0)

while True:
	_, frm = cap.read()

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27 or k == -1:
		old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
		cv2.destroyAllWindows()
		break

old_pts = np.array([ix,iy], dtype="float32").reshape(-1,1,2)
mask = np.zeros_like(frm)

while True:
	_, frame2 = cap.read()

	new_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

	new_pts,status,err = cv2.calcOpticalFlowPyrLK(old_gray, 
                         new_gray, 
                         old_pts, 
                         None, maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                         15, 0.08))

	cv2.circle(mask, (int(new_pts.ravel()[0]), int(new_pts.ravel()[1])), 2, (0,255,0), 2)
	combined = cv2.addWeighted(frame2, 0.7, mask, 0.3, 0.1)

	cv2.imshow("new win", mask)
	cv2.imshow("wind", combined)

	old_gray = new_gray.copy()
	old_pts = new_pts.copy()
    
    x1 = int(new_pts.ravel()[0]) - 100
    x2 = int(new_pts.ravel()[0]) + 100
    y1 = int(new_pts.ravel()[1]) - 100
    y2 = int(new_pts.ravel()[1]) + 100
    cropped_vid = frame2[x1:x2, y1:y2]
    cv2.imshow("Cropped Video", frame2)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
    

    



"""
while True:
    _, frame3 = cap.read()
    x1 = int(new_pts.ravel()[0]) - 100
    x2 = int(new_pts.ravel()[0]) + 100
    y1 = int(new_pts.ravel()[1]) - 100
    y2 = int(new_pts.ravel()[1]) + 100
    cropped_vid = frame3[x1:x2, y1:y2]
    frame3 = cv2.rectangle(frame3, [x1,y1], [x2,y2], [0,0,255], 2)
    cv2.imshow("Cropped Video", frame3)

    if cv2.waitKey(1) == 27:
        exit(0)
"""




