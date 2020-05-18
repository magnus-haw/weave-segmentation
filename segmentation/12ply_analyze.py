import cv2 as cv
from skimage import io
import numpy as np
import os, imageio
import matplotlib.pyplot as plt
from glob import glob

folder = "./ADEPT/"
imgRegx = folder + "adept12ply_raw.tif"
imgpaths = sorted(glob(imgRegx))
MAKE_TEST_SET = False
REASSEMBLE_OUTPUT = True
SHOW_OVERLAY = False

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

def grayMask(im):
    ret = im.copy()
    ret[im[:,:,0]>0] = 1
    ret[im[:,:,1]>0] = 2
    ret[im[:,:,2]>0] = 0
    return ret[:,:,0]

def reassemble(regx,shsq,imshape):
        sqpaths = sorted(glob(regx))
        ret = np.zeros(imshape,dtype=np.uint8)
        for path in sqpaths:
            
            ### parse filename
            folder, name, ext = splitfn(path)
            nlist = name.split("_")
            row,col = int(nlist[-2]),int(nlist[-1])
            ypx,xpx = row*shsq[0], col*shsq[1]

            ### offset border sqs
            if ypx+shsq[0] > imshape[0]:
                ypx = imshape[0] - shsq[0]
            if xpx+shsq[1] > imshape[1]:
                xpx = imshape[1] - shsq[1]

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

if MAKE_TEST_SET:

    ### Load metadata
    toppad,bottompad,leftpad,rightpad = 4,0,0,9

    ### Loop over images
    for n in range(1,nframes-1):

        ### crop non-segemented edges of masks & images
        imcrop = im[n-1:n+2,:,:]
        imcrop = np.moveaxis(imcrop, 0, -1)
        hc,wc,c= imcrop.shape
        
        full = np.zeros((304,384,3))
        full[:hc,:wc,:] = imcrop
        #print(full.shape)
        #print(magnus)

        ### Split images into smaller sq images
        imfolder = './12ply'
        cv.imwrite(imfolder+'/'+name+'_{:04d}'.format(n)+'.png',full)
                
if REASSEMBLE_OUTPUT:
    in_dir="./outputs/"
    labels = im.copy()*0

    for s in range(1,nframes-1):
        regex = in_dir + "adept12ply_raw_{:04d}.png".format(s)
        imgpaths = sorted(glob(regex))
        
        ret = cv.imread(imgpaths[0],1)
        image  = ret[:300,:375]
        gray = grayMask(image)
        labels[s,:,:] = gray
        
        cv.imwrite("./12ply_segmented/cnn_12ply_{:04d}.png".format(s),ret)

        if SHOW_OVERLAY:
            ### crop image to mask size
            imcrop = im[s,:,:]
            imcrop = cv.cvtColor(imcrop,cv.COLOR_GRAY2BGR)
            dst = overlayMask(imcrop,image)
            plt.title("cnn_12ply_{}".format(s))
            plt.imshow(dst)
            plt.show()
            
    imageio.mimwrite('adept12ply_labels.tiff',labels)
