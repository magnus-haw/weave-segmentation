import cv2 as cv
from skimage import io
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

folder = "./ADEPT/"
imgRegx = folder + "adept12ply_raw.tif"
imgpaths = sorted(glob(imgRegx))
sqshape = 64

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def processMask(im,th_bg=1,th_weft=60,color=False):
    ret = im.copy()
    warp = cv.inRange(ret,th_bg,th_weft)
    weft = cv.inRange(ret,th_weft,255)
    ret[warp>0] = 2
    ret[weft>0] = 1
    if color:
        ret *= 100
    return ret

### Read 12ply data cube file
path,name,ext = splitfn(imgpaths[0])
print(name)
ply = 12
im = io.imread(imgpaths[0])
nframes,h,w = im.shape

### Load metadata
toppad,bottompad,leftpad,rightpad = 0,0,0,0

### Loop over images
for n in range(1,nframes-1):

    ### crop non-segemented edges of masks & images
    imcrop = im[n-1:n+2,toppad:h-bottompad,leftpad:w-rightpad]
    imcrop = np.moveaxis(imcrop, 0, -1)
    hc,wc,c= imcrop.shape

    ### Split images into smaller sq images
    rows,cols = int(hc/sqshape), int(wc/sqshape)
    for j in range(0,rows):
        row = j*sqshape
        for k in range(0,cols):
            col = k*sqshape

            ### excise subimage
            imsq = imcrop[row:row+sqshape,col:col+sqshape,:]

            ### save subimages to training and validation folders
            imfolder = './12ply'
            cv.imwrite(imfolder+'/'+name+'_{}_{}_{}'.format(n,j,k)+'.png',imsq)
                
