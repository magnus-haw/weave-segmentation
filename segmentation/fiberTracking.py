### 3D fiber tracking
import cv2 as cv
from skimage import io
import numpy as np
import os, imageio
import matplotlib.pyplot as plt
from glob import glob
from GrabToes import GrabToes
import pickle

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def colorize(gray):
    h,w = gray.shape
    color = np.zeros((h,w,3))
    color[:,:,0] = (gray==1)*255.
    color[:,:,1] = (gray==2)*255.
    color[:,:,2] = (gray==0)*255.
    return color.astype(np.uint8)

# Marker labelling
def getWatershed(img,keypoints):
    h,w,c = img.shape
    mask = np.zeros((h,w),dtype=np.uint8)
    img2 = img.copy() # a copy of original image

    #set mask background to 1 & foreground to 0                       
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5), (2, 2))
    bl = cv.dilate(img2[:,:,0], element)
    gr,rd = img2[:,:,1],img2[:,:,2]
    rd[bl>0] = 0
    gr[bl>0] = 0
    img2[:,:,0],img2[:,:,1],img2[:,:,2] = bl,gr,rd
    plt.imshow(img2)
    plt.show()

    fgrd = cv.inRange(img2, (250, 0, 0), (256, 0, 0)).astype('uint8')
    bgrd = cv.inRange(img2, (0, 0, 0), (0, 255, 255)).astype('uint8')

    mask[bgrd==255]=1
    mask[fgrd==255]=0
    
    #plot markers with increasing values
    for i in range(0,len(keypoints)):
        value = i+2
        x,y = keypoints[i][0],keypoints[i][1]
        cv.circle(mask, (x,y), 2, value, -1)
    mmask = np.int32(mask)
    print(type(img2[0,0,0]))
    print(type(img2))
    markers = cv.watershed(img2,mmask)
    markers[markers == -1] = 0

    plt.imshow(markers)
    plt.show()
    return markers

### Read 12ply data cube file
folder = "./"
imgRegx = folder + "adept12ply_labels_full.tiff"
imgpaths = sorted(glob(imgRegx))
path,name,ext = splitfn(imgpaths[0])
LOAD = True
EDIT = False
im = io.imread(imgpaths[0])
nframes,h,w = im.shape

### Starting toe positions
if LOAD:
    fin = open('start407.pickle','rb')
    startpoints = pickle.load(fin)
    fin.close()
    frame = im[407,:,:352]
    color = colorize(frame)

    if EDIT:
        ### manually edit starting points
        startpoints = GrabToes().run(color,keypoints=startpoints)
        fout = open('start407.pickle','wb')
        pickle.dump(startpoints,fout)
        fout.close()
        
    segmented = getWatershed(color,startpoints)
    #locate centers
    #use centers for next slice    
    
else:
    ### Load frame
    frame = im[407,:,:352]
    color = colorize(frame)
    dst = (frame==1)*255
    dst = dst.astype(np.uint8)
    kernel =  cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)

    ### SimpleBlobDetector
    params = cv.SimpleBlobDetector_Params()
    params.blobColor =255

    # Change thresholds
    params.minThreshold = 250;
    params.maxThreshold = 255;
    params.thresholdStep = 2;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 55
    params.maxArea = 2500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.
    params.maxCircularity = 0.75

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.001

    # Min distance
    params.minDistBetweenBlobs = 3

    # Detect blobs
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dst)
    print(len(keypoints))
    im_with_keypoints = cv.drawKeypoints(dst, keypoints, dst, (0,150,0),
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp = [[int(k.pt[0]),int(k.pt[1])] for k in keypoints]

    ### manually edit starting points
    startpoints = GrabToes().run(color,keypoints=kp)
    fout = open('start407.pickle','wb')
    pickle.dump(startpoints,fout)
    fout.close()
    
### propagate, repeat


