import cv2 as cv
from skimage import io
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

folder = "./ADEPT/"
imgRegx = folder + "adept12ply_raw.tif"
imgpaths = sorted(glob(imgRegx))
MAKE_TRAINING_SET = False
REASSEMBLE_OUTPUT = True
SHOW_OVERLAY = True
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

### Read 12ply data cube file
path,name,ext = splitfn(imgpaths[0])
print(name)
ply = 12
im = io.imread(imgpaths[0])
nframes,h,w = im.shape

if MAKE_TRAINING_SET:

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
                
if REASSEMBLE_OUTPUT:
    out_dir="./outputs/"

    for s in range(0,nframes):
        regex = out_dir + "adept12ply_raw_{}_*.png".format(s)
        imgpaths = sorted(glob(regex))
        shsq,imshape = (64,64,3),(256,320,3)
        
        ret = reassemble(regex,shsq,imshape)
        cv.imwrite("./12ply_segmented/cnn_12ply_{:04d}.png".format(s),ret)

        if SHOW_OVERLAY:
            ### crop image to mask size
            imcrop = im[s,0:256,0:320]
            imcrop = cv.cvtColor(imcrop,cv.COLOR_GRAY2BGR)
            dst = overlayMask(imcrop,ret)
            plt.title("cnn_12ply_{}".format(s))
            plt.imshow(dst)
            plt.show()
