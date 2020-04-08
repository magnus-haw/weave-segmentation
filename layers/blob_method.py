import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob

folder = "./"
mask = folder + "ILPostTest-80um*.JPG"  # default

paths = glob(mask)

##def getBlobs(contours,minThreshold=1,maxThreshold=0):
    

for path in paths:
    rawimage = cv.imread(path,0)
##    plt.imshow(rawimage)
##    plt.show()
    
    roi0 = rawimage.copy()[41:110,356:454]
    roi = cv.medianBlur(roi0, 7)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(roi)
    roi_eq = cl1
##    plt.imshow(cl1)
##    plt.show()

    thresh = cv.adaptiveThreshold(roi_eq,255,
                               cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY,15,2)

    # noise removal
    kernel = np.ones((2,2),np.uint8)
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(thresh,cv.DIST_L2,3)
    ret, sure_fg = cv.threshold(dist_transform,0.18*dist_transform.max(),255,0)
    erode = cv.erode(sure_fg,kernel)
    erode = cv.dilate(erode,kernel)
    #opening = cv.morphologyEx(sure_fg,cv.MORPH_OPEN,kernel, iterations = 5)

    # contours
    output = erode.astype(np.uint8)
    contours,h = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(output, contours, -1, 255, -1)
    cnts = sorted(contours, key=cv.contourArea)
    

    plt.imshow(roi0)

    plt.figure(10)
    plt.imshow(output)
    #plt.show()

    # make mask
    mask = cv.cvtColor(output, 1, cv.COLOR_GRAY2BGR)

    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    params.blobColor =255
    
    # Change thresholds
    params.minThreshold = 50;
    params.maxThreshold = 150;
    params.thresholdStep = 10;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 150

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.25

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.001

    params.minDistBetweenBlobs = 8

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)
    print(len(keypoints))

    im_with_keypoints = cv.drawKeypoints(roi0, keypoints, roi, (0,150,0),
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(9)
    plt.imshow(im_with_keypoints)
    plt.show()
