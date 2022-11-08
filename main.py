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

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
 
# Set up output video
out = cv2.VideoWriter('video_out.mp4', fourcc, fps=cap.get(cv2.CAP_PROP_FPS), frameSize= (width, height))

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


#explicitly shows all the different transforms of dx,dy,da
all_dx = transforms[:,0]
all_dy = transforms[:,1]
all_da = transforms[:,2]
trajectory = np.cumsum(transforms,axis=0)
Smoothing_Radius = 50

fr = np.arange(0,num_frames-1,1)



#creating a function for moving averages to smooth out the curve 
def MovingAverage(c,r):
    window = 2*r*+1
    #filter
    filter = np.ones(window)/window
    #add padding
    c_pad = np.lib.pad(c,(r,r),"edge")
    #applying convolution 
    smoothed_c = np.convolve(c_pad,filter,mode="same")
    #removes padding going from r to -r
    smoothed_c = smoothed_c[r:-r]
    return smoothed_c

def SmoothTrajectory(trajectory):
    global Smoothing_Radius
    smoothed_trajectory = np.copy(trajectory)
    #filters x,y and angle of the curve
    for i in range(3):
        smoothed_trajectory[:,i] = MovingAverage(trajectory[:,i],r = Smoothing_Radius)

    return smoothed_trajectory

def borderfixing(frame):
    s = frame.shape
    #scale the image 4% without moving the centre 
    T = cv2.getRotationMatrix2D((s[1]/2,s[0]/2),0,1.04)
    frame = cv2.warpAffine(frame,T,(s[1],s[0]))
    return frame



difference = SmoothTrajectory(trajectory) - trajectory
# Calculate newer transformation array
transforms_smooth = transforms + difference



#setting the video's frame back to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES,0)

for i in range(num_frames-2):
    ret,frames = cap.read()
    if not ret: 
        break
    
    # Extract transformations from the new transformation array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]
    
    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy
    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (width,height))
    
    # Fix border artifacts
    frame_stabilized = borderfixing(frame_stabilized) 
    
    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (width,height))

	# Fix border artifacts
    frame_stabilized = borderfixing(frame_stabilized) 

	# Write the frame to the file
    frame_out = cv2.hconcat([frame, frame_stabilized])

	# If the image is too big, resize it.
    if(np.array(frame_out).shape[1] > 1920): 
        frame_out = cv2.resize(frame_out, (int(np.array(frame_out).shape[1]/2), int(np.array(frame_out).shape[0]/2)))


    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)
out.write(frame_out)
        
# Release video
cap.release()
out.release()

#creates figure for the transforms changes in pixles
plt.figure(figsize=[8,6])
plt.ylabel("delta pixels")
plt.xlabel("frames")
plt.plot(fr,all_dx,color="blue",label="dx")
plt.plot(fr,all_dy,color="green",label="dy")
plt.plot(fr,all_da,color="red",label="da")
plt.legend()

#smooth transforms 
plt.figure(figsize=[8,6])
plt.title("Smoothed out transforms dx")
plt.ylabel("delta pixels")
plt.xlabel("frames")
plt.plot(fr,transforms_smooth[:,0])

#creates figure for the trajectory of motion
plt.figure(figsize=[8,6])
plt.ylabel("delta pixels")
plt.xlabel("frames")
plt.plot(fr,trajectory[:,0],color="blue",label="x trajectory")
plt.plot(fr,trajectory[:,1],color="green",label="y trajectory")
plt.plot(fr,trajectory[:,2],color="red",label="a trajectory")
plt.legend()



YT_dx = np.fft.fft(all_dx)
YT_dxr = np.fft.rfft(all_dx)
YT_dy = np.fft.fft(all_dy)
YT_da = np.fft.fft(all_da)
XT = np.fft.fftfreq(num_frames-1)
XTr = np.fft.rfftfreq(num_frames-1)
plt.figure(figsize=[8,6])
plt.title("Fourier Transform of dx")
plt.ylabel("Amplitude")
plt.xlabel("Relative Frequencies")
plt.plot(XT,YT_dx)
plt.figure(figsize=[8,6])
plt.title("Fourier Transform of the dx  only real numbers")
plt.ylabel("Amplitude")
plt.xlabel("Relative Frequencies")
plt.plot(XTr,YT_dxr)




plt.show()





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
