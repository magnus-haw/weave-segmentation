import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def reassemble(regx,shsq,imshape):
        sqpaths = sorted(glob(regx))
        ret = np.zeros(imshape,dtype=np.uint8)
        for path in sqpaths:

            ### parse filename
            folder, name, ext = splitfn(path)
            nlist = name.split("_")
            row,col = int(nlist[-2]),int(nlist[-1])
            ypx,xpx = row*shsq[0], col*shsq[1]

            ### Read image
            im = cv.imread(path,1)

            ### paste image into final image
            ret[ypx:ypx+shsq[0],xpx:xpx+shsq[1],:] = im.astype(np.uint8)
        return ret
    
def overlayMask(im,mask,alpha=.75):
    beta = (1.0 - alpha)
    dst = cv.addWeighted(im, alpha, mask, beta, 0.0)
    return dst

toppad = 40
leftpad = 75
rightpad= 176
bottompad= 34

for i in [0,1]:
    regx = "./outputs/ms_8ply_000{}*.png".format(i)
    shsq,imshape = (64,64,3),(512,1024,3)
    
    ret = reassemble(regx,shsq,imshape)
    cv.imwrite("cnn_ms_8ply_000{}.png".format(i),ret)
##    plt.imshow(ret)
##    plt.show()

    ### crop image to mask size
    im = cv.imread("./ms_8ply_000{}.tif".format(i),1)
    h,w,c = im.shape
    imcrop = im[toppad:h-bottompad,leftpad:w-rightpad,:]

    dst = overlayMask(imcrop,ret)
    plt.title("ms_8ply_000{}".format(i))
    plt.imshow(dst)
    plt.show()
