import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob

folder = "./"
mask = folder + "ms_8ply_000?.tif"  # default

paths = glob(mask)

for path in paths:
    roi = cv.imread('outplane_1.tif',0)
    target = cv.imread(path)

    # plot image, roi
    plt.subplot(211), plt.imshow(roi)
    plt.subplot(212), plt.imshow(target)
    plt.show()
    
    # calculating object histogram
    roihist = cv.calcHist([roi],None, None, [256], (0, 256) )
    plt.plot(roihist)
    plt.show()
    
    # normalize histogram and apply backprojection
    cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
    dst = cv.calcBackProject([target],[0],roihist,[0,256],1)
    plt.imshow(dst)
    plt.show()
    
    # Now convolute with circular disc
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    cv.filter2D(dst,-1,disc,dst)

    plt.imshow(dst)
    plt.show()

    # threshold and binary AND
    ret,thresh = cv.threshold(dst,50,255,0)
    thresh = cv.merge((thresh,thresh,thresh))
    res = cv.bitwise_and(target,thresh)
    res = np.vstack((target,thresh,res))
    cv.imwrite('res.jpg',res)
