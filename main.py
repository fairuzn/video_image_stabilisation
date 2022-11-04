#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
path = 'Simple1.mp4'

cap = cv2.VideoCapture(path)
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

#closes the object 
cap.release()
#closes the window for the video 
cv2.destroyAllWindows()
#numpy array of the frames, the last value is a none value as ret is false
f = np.array(frame_list,dtype="object")
height = f[0].shape[0]
width = f[0].shape[1]

plt.imshow(f[0],aspect= "auto")

#75,60,82
#creating listing for pixel coordinate 
test = np.zeros(shape=(height,width,3))
pixeldict = {}
for i in range(height):
    for j in range(width):
        #the notation is as f[frame][height,width]
        if f[0][i,j][0] <= 116 and f[0][i,j][1] <= 120 and f[0][i,j][2] >= 82:
            pixeldict[(i, j)] = f[0][i][j]

        else: 
            continue



"""for i in range(len(pixeldict)):
    if (i,j) == list(pixeldict.keys())[i]:
        test[i,j] = pixeldict[(i,j)]
    else: 
        continue"""

pixelco = list(pixeldict.keys())
print(pixelco[0][0])
for i in range(len(pixelco)):
    test[pixelco[i][0],pixelco[i][1]] = pixeldict[(pixelco[i])]

print("rgb at 0,0:" ,pixeldict[(500,500)])
plt.figure("testing")
plt.imshow(test)
plt.show()

# %%
