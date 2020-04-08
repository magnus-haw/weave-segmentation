import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from glob import glob

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def build_filters(ksize=9,nangles=12,sigma=3,lmbda=3.5,gamma=.46,psi=0):
    filters = []
    for theta in np.linspace(0, np.pi, nangles):
        kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lmbda, gamma, psi,
                             ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def textureFilter(img,params=None):
    if params is None:
        x= [9,12,2.906,3.485,.47]
    else:
        x = params
    filters = build_filters(ksize=int(x[0]),nangles=int(x[1]),sigma=x[2],
                            lmbda=x[3],gamma=x[4],psi=0)
    res1 = process(img, filters)
    
    ### noise removal
    texture = cv.medianBlur(res1, 5, 0)
    stexture = cv.GaussianBlur(texture, (5, 5), 0)

    ### Threshold
    th1 = cv.inRange(stexture,200,255,cv.THRESH_BINARY)
    th1 = cv.medianBlur(th1, 11, 0)
    return th1

def fitTextureParams(filemask):
    paths = glob(filemask)
    fits = []
    for path in paths:

        pth, name, ext = splitfn(path)

        ### Read in images
        img = cv.imread(name+'.tif',0)
        mask = cv.imread(name+'.png')

        ### Compare with mask
        b,g,r = cv.split(mask)
        bpx,gpx,rpx = cv.countNonZero(b),cv.countNonZero(g),cv.countNonZero(r)

        def getContrast(x):
            ### Apply Gabor filters
            res1 = textureFilter(img,x)
            
            bres = cv.bitwise_and(res1,res1,mask=b)
            gres = cv.bitwise_and(res1,res1,mask = g)
            rres = cv.bitwise_and(res1,res1,mask = r)

            bavg,gavg = bres.sum()/bpx, gres.sum()/gpx
            return 1/(bavg - gavg)

        x0 = np.array([9,12,2.906,3.485,.47])
        minmodel = minimize(getContrast, x0, method='nelder-mead',
                            options={'xatol': 1e-8, 'disp': True})
        print(minmodel.x)
        fits.append(minmodel.x)
    return fits

PLOT=True
if __name__ == "__main__":

    filemask = "ms_8ply_000?.tif"
    paths = glob(filemask)

    for path in paths:
        pth, name, ext = splitfn(path)
        print(name)

        ### Read in image
        img = cv.imread(name+'.tif',0)

        #####################################################################
        ###Filter for background
        roi = cv.imread('outplane_1.tif',0)
        # calculating object histogram
        roihist = cv.calcHist([roi],None, None, [256], (0, 256) )
        bgrd = cv.calcBackProject([img],[0],roihist,[0,256],1)
        # Now convolute with circular disc
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        cv.filter2D(bgrd,-1,disc,bgrd)

        # threshold and binary AND
        thresh = cv.inRange(bgrd,0,50,0)
        ### noise removal
        th_bgrd = cv.medianBlur(thresh, 11, 0)

        if PLOT:
            plt.imshow(th_bgrd)
            plt.show()
        
        #####################################################################
        ### Filter for warp
        x = np.array([9,12,2.9,3.48,.4])
        th1 = textureFilter(img,x)

        ### Erode & dialate to remove fluff
        kernel1 = np.ones((5,5), dtype=np.uint8)
        eroded = cv.erode(th1, kernel1)

        kernel2 = np.ones((3,3), dtype=np.uint8)
        th_warp = cv.dilate(eroded, kernel2)

        if PLOT:
            plt.imshow(th_warp)
            plt.show()
        #####################################################################
        ### Filter for weft
        th_weft = cv.bitwise_and(255-th_warp,255-th_bgrd)
        ### noise removal
        th_weft = cv.medianBlur(th_weft, 11, 0)

        if PLOT:
            plt.imshow(th_weft)
            plt.show()

##        #####################################################################
##        ### Compare with mask
##        mask = cv.imread(name+'.png')
##        b,g,r = cv.split(mask)
##        bpx,gpx,rpx = cv.countNonZero(b),cv.countNonZero(g),cv.countNonZero(r)
##        compare = cv.bitwise_and(th_weft,g,mask=g)
##        
##        print(compare.sum()/gpx/255)
##        plt.figure()
##        plt.subplot(211);plt.imshow(th_weft)
##        plt.subplot(212);plt.imshow(g)
##        plt.show()

        ### export to file
        h,w = np.shape(img)
        new_img = np.zeros([h,w,3])

        new_img[:,:,0] = th_warp
        new_img[:,:,1] = img
        new_img[:,:,2] = th_bgrd
        
        cv.imwrite(name+"_preproc"+'.png', new_img)

        if PLOT:
            plt.imshow(new_img.astype(int))
            plt.show()
        

