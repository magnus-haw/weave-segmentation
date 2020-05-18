import numpy as np
import cv2 as cv
import os, pickle
import matplotlib.pyplot as plt
from glob import glob
from GrabToes import GrabToes
from DBSCAN import getDBSCAN,getClusters
from template_method import findBlobs
from gettemps import GetTemplates

folder = "./"
prefix = "A1-10mil-57um"
r1,r2 = 30,406
minDist = 6
eps=0.35
minArea=50
maxArea=150
threshold=140

##prefix = "ILPreTest-40um-mid"
##r1,r2 = 61,240
##minDist = 15
##eps = 0.5
##minArea=150
##maxArea=500
##threshold=90

##prefix = "RLPreTest-40um-mid"
##r1,r2 = 162,305
##minDist = 4
##eps = 0.31
##minArea=90
##maxArea=160
##threshold=90

fnames = folder + prefix + ".JPG"  # default
paths = glob(fnames)

NEW_TEMPLATES = False

### Extract toe templates
if NEW_TEMPLATES:
    img=cv.imread(prefix+".JPG",1)
    gtemp = GetTemplates()
    gtemp.run(img,prefix)
folder = "./templates/"
toes = glob(folder + prefix + "*toe*.jpg")

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

toe_temps=[]
for toe in toes:
    toe_temps.append(cv.imread(toe,0))
temp_areas = [t.sum()/255 for t in toe_temps]

rawimage = cv.imread(paths[0],0)
plt.imshow(rawimage)
plt.show()

### Region of interest
roi0 = rawimage.copy()[r1:r2,:]

if input("Load or detect? (l/d) ") == 'l':
##if True:
    fin = open(prefix+'_keypoints.pickle','rb')
    kp = pickle.load(fin)
    fin.close()

else:
    kp = findBlobs(roi0, toe_temps, minArea=10, maxArea=100,
          minDist=minDist,minThresh=100,thresholdStep=2,
          plot=False)
    
# ##################################################################
# Interactive manual review stage
frame = cv.cvtColor(roi0, cv.COLOR_GRAY2BGR)
gt = GrabToes()
finalpoints,imarked = gt.run(frame,keypoints = kp)

if input("save? (y/n) ") == 'y':
##if 'n' == 'y':
    fout = open(prefix+'_keypoints.pickle','wb')
    pickle.dump(finalpoints,fout)
    fout.close()

### Figure amalgamation
fig = plt.figure(0)
ax0 = plt.subplot(241)
plt.imshow(imarked)
plt.title(prefix+ " Tow Identification")

# ##################################################################
# Get toe dimension stats
ax5 = plt.subplot(245)
ret,fitImg = gt.getTowDimensions(plot=True,minArea=minArea,maxArea=maxArea,
                                 threshold=threshold)
plt.title("Ellipse fits")

minoraxis,majoraxis = ret[:,0],ret[:,1]
angle = ret[:,2]-90
ax1 = plt.subplot(246)
plt.hist(angle)
print("ellipse angle (deg): ",np.mean(angle))
print("ellipse angle stdev (px): ",np.std(angle))
ax1.annotate("mean %.1f\nstdev %.1f"%(np.mean(angle),np.std(angle)),
            xy=(.2, .1),  xycoords='axes fraction',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )
plt.title("Angle from horizontal (deg)")
ax2 = plt.subplot(247)
plt.hist(majoraxis)
print("Major axis mean (px): ",np.mean(majoraxis))
print("Major axis stdev (px): ",np.std(majoraxis))
ax2.annotate("mean %.1f\nstdev %.1f"%(np.mean(majoraxis),np.std(majoraxis)),
            xy=(.2, .1),  xycoords='axes fraction',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )
plt.title("Tow Major Axis (px)")
ax3 = plt.subplot(248)
plt.hist(minoraxis)
print("Minor axis mean (px): ",np.mean(minoraxis))
print("Minor axis stdev (px): ",np.std(minoraxis))
ax3.annotate("mean %.1f\nstdev %.1f"%(np.mean(minoraxis),np.std(minoraxis)),
            xy=(.2, .1),  xycoords='axes fraction',
            xytext=(0.8, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )
plt.title("Tow Minor Axis (px)")

# ##################################################################
# Compute DBSCAN, get labels
ax4 = plt.subplot(242)
##plt.figure()
##ax = plt.subplot(111,aspect='equal')
labels,n = getDBSCAN(finalpoints, eps=eps, min_samples=2, plot=True)
plt.title("Identified columns %i"%n)

# ##################################################################
# Sort points into clusters
final = getClusters(finalpoints,labels,n,plot=False)

# ##################################################################
# Get vertical spacing stats
vdist = []
for cluster in range(0,n):
    c = final[cluster]
    for i in range(1,len(c)):
        vdist.append(abs(c[i][1]-c[i-1][1]))
vdist = np.array(vdist)
print("Vdist mean (px): ",np.mean(vdist))
print("Vdist stdev (px): ",np.std(vdist))
ax6 = plt.subplot(243)
plt.hist(vdist,bins=range(min(vdist),max(vdist)+1 ))
plt.title("Vertical spacing (px)")
ax6.annotate("mean %.1f\nstdev %.1f"%(np.mean(vdist),np.std(vdist)),
            xy=(.2, .1),  xycoords='axes fraction',
            xytext=(0.5, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )

# ##################################################################
# Get horizontal spacing stats
hdist = []
for cluster in range(1,n):
    c1,c2 = final[cluster-1],final[cluster]
    if len(c1) > len(c2):
        bc,sc = c1,c2
    else:
        bc,sc = c2,c1

    for i in range(0,len(sc)):
        vpos = sc[i][1]
        vdiff = abs(vpos-bc[:,1])
        ind = np.argmin(vdiff)
        hdist.append(abs(sc[i][0]-bc[ind][0]))
        
hdist = np.array(hdist)
print("Hdist mean (px): ",np.mean(hdist))
print("Hdist stdev (px): ",np.std(hdist))
ax7 = plt.subplot(244)
plt.hist(hdist,bins=range(min(hdist),max(hdist)+1 ))
plt.title("Horizontal spacing (px)")
ax7.annotate("mean %.1f\nstdev %.1f"%(np.mean(hdist),np.std(hdist)),
            xy=(.2, .1),  xycoords='axes fraction',
            xytext=(0.5, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )
plt.show()

