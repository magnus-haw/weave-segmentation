### 3D fiber tracking
import cv2 as cv
from skimage import io
import numpy as np
import os, imageio
import matplotlib.pyplot as plt
from glob import glob
from GrabToes import GrabToes

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

### Read 12ply data cube file
folder = "./"
imgRegx = folder + "adept12ply_labels_full.tiff"
imgpaths = sorted(glob(imgRegx))
path,name,ext = splitfn(imgpaths[0])
print(name)

im = io.imread(imgpaths[0])
nframes,h,w = im.shape

### Load frame
frame = im[407,:,:352]
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

# Min distance
params.minDistBetweenBlobs = 3

# Detect blobs
detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(dst)
print(len(keypoints))
im_with_keypoints = cv.drawKeypoints(dst, keypoints, dst, (0,150,0),
                                      cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp = [[int(k.pt[0]),int(k.pt[1])] for k in keypoints]

color = colorize(frame)
##plt.imshow(color)
##plt.show()

### manually edit starting points
startpoints = GrabToes().run(color,keypoints=kp)

### propagate, repeat
