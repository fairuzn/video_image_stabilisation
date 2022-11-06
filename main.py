#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
path = 'Simple1.mp4'

#create video object
cap = cv2.VideoCapture(path)
#number of frames 
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
#reads if it's ongoing and frames
ret, frame = cap.read()
#converts first frame to grayscale
last_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#an array to store the transformation occuring at pixels
#3 dimensions to account for dx,dy and da where da is the rotation
transforms = np.zeros((num_frames-1, 3), np.float32) 

#loop to get the movment of pixels between frames
#for i in range(num_frames-2):
i= 0
for i in range(num_frames-2): 
    ret, frame = cap.read()
    last_points = cv2.goodFeaturesToTrack(last_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
    i+= 1
    if not ret: 
        break
    
    #convertion to gray scale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points)
    points,status, err = cv2.calcOpticalFlowPyrLK(last_gray, gray, last_points, None) 

    # Sanity check
    assert last_points.shape == points.shape 

    # Filter only valid points
    index = np.where(status==1)[0]
    last_points = last_points[index]
    points = points[index]
    
    #Find transformation matrix
    matrix = cv2.estimateAffine2D(last_points, points)[0]


    # traslation of points
    dx = matrix[0,2]
    dy = matrix[1,2]

    #rotation angle of point
    #matrix[1,0] = opp = x and  matrix[0,0] = adj = y
    da = np.arctan2(matrix[1,0], matrix[0,0])

    # Store transformation
    transforms[i] = [dx,dy,da]

    # Move to next frame
    prev_gray = gray
 
    print("Frame: " + str(i) +  "/" + str(num_frames) + " -  Tracked points : " + str(len(last_points)))





"""frame_list = []
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
        if f[0][i,j][0] <= 105 and f[0][i,j][1] <= 105 and f[0][i,j][2] >= 10:
            pixeldict[(i, j)] = f[0][i][j]

        else: 
            continue"""



"""for i in range(len(pixeldict)):
    if (i,j) == list(pixeldict.keys())[i]:
        test[i,j] = pixeldict[(i,j)]
    else: 
        continue"""

"""pixelco = list(pixeldict.keys())
print(pixelco[0][0])
for f in range(len(frame_list)):
    for i in range(len(pixelco)):
        test[pixelco[i][0],pixelco[i][1]] = pixeldict[(pixelco[i])]"""


"""#print("rgb at 0,0:" ,pixeldict[(500,500)])
plt.figure("testing")
plt.imshow(test)
plt.show()"""

# %%
