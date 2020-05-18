import numpy as np
import cv2 as cv
import os, pickle
import matplotlib.pyplot as plt
from glob import glob
from GrabToes import GrabToes
from DBSCAN import getDBSCAN 

##folder = "./"
##prefix = "A1-10mil-57um"
##fnames = folder + prefix + ".JPG"  # default
##r1,r2 = 25,143
##
##r1,r2 = 30,406
##
##paths = glob(fnames)
##
##folder = "./templates/"
##toes = glob(folder + prefix + "*toe*.jpg")
##toe_temps=[]
##for toe in toes:
##    toe_temps.append(cv.imread(toe,0))

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def multiTemplateCorr(src, temps, srcThresh=95, iters=1):
    out = np.ones(np.shape(src))
    for temp in temps:
        #edge padding sizes
        h,w = np.shape(temp)
        top,left = int((h-1)/2),int((w-1)/2)
        bottom,right = h-1-top,w-1-left
        mt = cv.matchTemplate(src.astype(np.float32), temp.astype(np.float32), cv.TM_CCORR_NORMED)

        out[top:-bottom,left:-right] *= mt
    out[src<srcThresh] = 0
    out = cv.normalize(out,None,0,255,cv.NORM_MINMAX)
    out[out>=250] = 0

    return out.astype(np.uint8)

def postProcessCorr(src,iterations=3,CLAHE=True):
    # renormalize histogram
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(0,iterations):    
        out = (src.astype(np.float32))**2
        out = cv.normalize(out,None,0,255,cv.NORM_MINMAX)
        out = clahe.apply(out.astype(np.uint8))
    out = cv.equalizeHist(out.astype(np.uint8))
    return out

def findBlobs(roi0, toe_temps, minArea=5, maxArea=50,
              minDist=6,minThresh=100,thresholdStep=2,
              plot=False):
    ### Smoothing
    roi = cv.medianBlur(roi0, 5)
    
    ### Adaptive contrast filter
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(roi)

    ### Apply template correlation 
    dst = multiTemplateCorr(roi_eq, toe_temps)
    dst = postProcessCorr(dst)

    if plot:
        plt.imshow(dst)
        plt.show()

    ### Apply thresholding
    ret,thresh = cv.threshold(dst,200,255,cv.THRESH_BINARY)

    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    params.blobColor =255
    
    # Change thresholds
    params.minThreshold = minThresh;
    params.maxThreshold = 255;
    params.thresholdStep = thresholdStep;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = minArea
    params.maxArea = maxArea

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.25

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.001

    # Min distance
    params.minDistBetweenBlobs = minDist

    # Detect blobs
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dst)
    print("keypoints found:",len(keypoints))
    im_with_keypoints = cv.drawKeypoints(roi0, keypoints, roi0, (0,150,0),
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

##    # Write to file
##    folder,name,ext = splitfn(path)
##    cv.imwrite(folder+'/'+name+'_marked'+'.png',im_with_keypoints)
##    print(folder+'/'+name+'_marked'+'.png')

    # Plot marked locations
    plt.figure(9)
    plt.imshow(im_with_keypoints)
    plt.show()

    kp = [[int(k.pt[0]),int(k.pt[1])] for k in keypoints]
    
    return kp
