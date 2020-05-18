#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using Watershed algorithm.

This sample shows interactive image segmentation using watershed algorithm.

USAGE:
    python GrabToes.py <filename>

README FIRST:
    One window will show up for input.

    Double click left mouse button to add a toe
    Double click middle mouse button to remove a toe

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 'esc' - To return the results
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys



class GrabToes(object):
    BLUE = [255,0,0]        # rectangle color
    RED = [0,0,255]         # PR BG
    GREEN = [0,255,0]       # PR FG
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}

    # setting up flags
    value = DRAW_FG         # drawing initialized to FG
    thickness = 3           # brush thickness

    def get_nearest(self,x,y):
        mindist = 10000
        minInd =0
        for i in range(0,len(self.keypoints)):
            dist = np.sqrt((self.keypoints[i][0]-x)**2 + (self.keypoints[i][1]-y)**2)
            if dist < mindist:
                ret = self.keypoints[i]
                mindist = dist
                minInd = i
        return ret,minInd

    def onmouse(self, event, x, y, flags, param):
        # draw keypoints
        if event == cv.EVENT_LBUTTONDBLCLK:
            print("place keypoint")
            self.keypoints.append([x,y])
            print(x,y)
            cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
            cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)
        if event == cv.EVENT_MBUTTONDBLCLK:
            print("remove keypoint")
            ret,i = self.get_nearest(x,y)
            self.keypoints.pop(i)
            self.replot_keypoints()
            print(ret)

    def reset(self):
        self.img = self.img2.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
        self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown

    def replot_keypoints(self):
        self.reset()
        for kp in self.keypoints:
            x,y = kp[0],kp[1]
            cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
            cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

    def run(self,fn,keypoints=None):
        # Loading images
        self.img = fn
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8)
        
        if keypoints is None:
            self.keypoints = []
        else:
            self.keypoints = keypoints

        self.img2 = self.img.copy() # a copy of original image
        self.replot_keypoints()
        
        # input and output windows
        cv.namedWindow('input')
        cv.setMouseCallback('input', self.onmouse)
        cv.moveWindow('input', self.img.shape[1]+10,90)

        print("""Double click left mouse button to add a toe
    Double click middle mouse button to remove a toe

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 'esc' - To return the results""")

        while(1):
            #cv.imshow('output', self.output)
            cv.imshow('input', self.img)
            k = cv.waitKey(1)

            # key bindings
            if k == 27:         # esc to exit
                break
            elif k == ord('r'): # reset everything
                print("resetting \n")
                self.keypoints=[]
                self.reset()
            elif k == ord('n'): # segment the image
                self.watershedSegmentation(plot=True)

        print('Done')
        cv.destroyAllWindows()
        return self.keypoints,self.img

    def watershedSegmentation(self,plot=False,threshold=140):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        ret,thresh = cv.threshold(gray,threshold,255,cv.THRESH_BINARY)
        ret,self.bgrd = cv.threshold(gray,threshold,255,cv.THRESH_BINARY_INV)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        self.fgrd = cv.morphologyEx(thresh, cv.MORPH_OPEN,  kernel)
        ret,self.fgrd = cv.threshold(self.fgrd,threshold,1,cv.THRESH_BINARY)
        self.thresh = cv.cvtColor(self.fgrd, cv.COLOR_GRAY2BGR)
        print(""" segmenting... """)
        try:
            # Marker labelling
            ret, markers = cv.connectedComponents(self.mask)
            unknown = cv.subtract(self.fgrd,self.mask)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            # Now, mark the region of unknown with zero
            markers[unknown==1] = 0
            
            markers = cv.watershed(self.thresh,markers)
            markers[markers == -1] = 0
            self.markers = markers
            if plot:
                plt.imshow(markers)            
        except:
            import traceback
            traceback.print_exc()

    def getTowDimensions(self,plot=False,minArea=55,maxArea=190,threshold=140):
        ret = []
        self.watershedSegmentation(threshold=threshold)
        img = self.markers
        new = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        print(""" fitting ellipses... """)
        for i in range(2,img.max()+1):
            mask = (img==i).astype(np.uint8)
            contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            cnt = contours[0]
            area = cv.contourArea(cnt)
            if area > minArea and area < maxArea:
                ellipse = cv.fitEllipse(cnt)
                ret.append(np.array([ellipse[1][0],ellipse[1][1],ellipse[2]]))
                cv.ellipse(new,ellipse,(0,255,0),2)
                if plot:
                    plt.imshow(new)
        return np.array(ret),new


if __name__ == '__main__':
    print(__doc__)
    frame = cv.imread("A1-10mil-57um.JPG",1)
    #frame = cv.imread("coins.png",1)
    gt = GrabToes()
    gt.run(frame)
