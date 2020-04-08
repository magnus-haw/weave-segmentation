import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

folder = "./"
imgRegx = folder + "ms_8ply_000?.tif"  
maskRegx = folder + "ms_8ply_000?.png"  
metaRegx = folder + "ms_8ply_000?.txt"

imgpaths = sorted(glob(imgRegx))
maskpaths = sorted(glob(maskRegx))
metapaths = sorted(glob(metaRegx))

### input params
toppad = 40
leftpad = 75
rightpad= 176
bottompad= 34

sqshape = 64

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def processMask(im):
    ret = im.copy()
    ret[im[:,:,0]>0] = 2
    ret[im[:,:,1]>0] = 1
    ret[im[:,:,2]>0] = 0
    return ret

### Loop over images & masks
for i in range(0,len(imgpaths)):
    path,name,ext = splitfn(imgpaths[i])

    ### split test and training sets
    if i==0 or i==1:
        flag = 'test'
    else:
        flag = 'train'

    ### crop non-segemented edges of masks & images
    im = cv.imread(imgpaths[i],0)
    mask = cv.imread(maskpaths[i],1)
    h,w,c = mask.shape
    imcrop = im[toppad:h-bottompad,leftpad:w-rightpad]
    mcrop = mask[toppad:h-bottompad,leftpad:w-rightpad,:]

    ### Load meta data with errors in masks
    meta = np.loadtxt(metapaths[i],delimiter=',')

    ### Split images into smaller sq images
    rows,cols = int(512/sqshape), int(1024/sqshape)
    for j in range(0,rows):
        row = j*sqshape
        for k in range(0,cols):
            col = k*sqshape

            ### excise subimage
            imsq = imcrop[row:row+sqshape,col:col+sqshape]
            msq  =  mcrop[row:row+sqshape,col:col+sqshape]

            ### check if mask has error
            err = False
            for pt in meta:
                r,c = pt
                if  row < r < row+sqshape and col <= c <= col+sqshape:
                    err = True
                    break
            if err and flag == 'train':
                mcrop[row:row+sqshape,col:col+sqshape] = 0
                
            else:
                ### save subimages to folders
                if flag == 'train':
                    imfolder = './train_frames'
                    mfolder = './train_masks'
                    pmask = processMask(msq)
                    cv.imwrite( mfolder+'/'+name+'_{}_{}'.format(j,k)+'.png',pmask)
                else:
                    imfolder = './val_frames'
                    mfolder = './val_masks'
                    cv.imwrite( mfolder+'/'+name+'_{}_{}'.format(j,k)+'.png',msq)
                cv.imwrite(imfolder+'/'+name+'_{}_{}'.format(j,k)+'.png',imsq)
                

##    dst = overlayMask(imcrop,mcrop)
##    plt.title(name)
##    plt.imshow(dst)
##    plt.show()
    

