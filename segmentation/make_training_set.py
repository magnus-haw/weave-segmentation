import cv2 as cv
from skimage import io
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob

folder = "./ADEPT/"
imgRegx = folder + "adept?ply_raw_05.tif"  
maskRegx = folder + "adept?ply_tagged_05.tif"
metaRegx = folder + "adept?ply_meta_05.csv"

imgpaths = sorted(glob(imgRegx))
maskpaths = sorted(glob(maskRegx))
metapaths = sorted(glob(metaRegx))

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

### Loop over different ply data cubes
for i in range(0,len(imgpaths)):
    path,name,ext = splitfn(imgpaths[i])
    print(name)
    ply = int(name[5])
    
    ### Load data
    im = io.imread(imgpaths[i])
    mask = io.imread(maskpaths[i])
    nframes,h,w = im.shape

    ### Load metadata
    meta = np.loadtxt(metapaths[i],delimiter=',',dtype=np.uint8)
    toppad,bottompad,leftpad,rightpad = meta

    ### Loop over images
    for n in range(1,nframes-1):

        ### crop non-segemented edges of masks & images
        imcrop = im[n-1:n+2,toppad:h-bottompad,leftpad:w-rightpad]
        mcrop = mask[n,toppad:h-bottompad,leftpad:w-rightpad]
        imcrop = np.moveaxis(imcrop, 0, -1)
        hc,wc,c= imcrop.shape

        thresh = 40
        if ply==8:
            thresh = 60

        ### split test and training sets
        if np.random.random_sample() > .8:
            flag = 'test'
        else:
            flag = 'train'

        ### Split images into smaller sq images
        rows,cols = int(hc/sqshape), int(wc/sqshape)
        for j in range(0,rows):
            row = j*sqshape
            for k in range(0,cols):
                col = k*sqshape

                ### excise subimage
                imsq = imcrop[row:row+sqshape,col:col+sqshape,:]
                msq  =  mcrop[row:row+sqshape,col:col+sqshape]

                ### save subimages to training and validation folders
                if flag == 'train':
                    imfolder = './train_frames'
                    mfolder = './train_masks'
                    pmask = processMask(msq,th_weft=thresh)
                    cv.imwrite( mfolder+'/'+name+'_{}_{}_{}'.format(n,j,k)+'.png',pmask)
                else:
                    imfolder = './val_frames'
                    mfolder = './val_masks'
                    pmask = processMask(msq,th_weft=thresh,color=True)
                    cv.imwrite( mfolder+'/'+name+'_{}_{}_{}'.format(n,j,k)+'.png',pmask)
                cv.imwrite(imfolder+'/'+name+'_{}_{}_{}'.format(n,j,k)+'.png',imsq)
                
