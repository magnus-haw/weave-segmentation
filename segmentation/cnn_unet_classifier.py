from keras.models import Input,load_model
from keras.layers import Dropout,concatenate,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D

from keras_segmentation.predict import predict_multiple
from keras_segmentation.train import find_latest_checkpoint
from glob import glob
import os

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

TRAIN = False
LOAD = True
APPLY = True
input_height,input_width = 128,128
n_classes = 3
epochs = 5
ckpath = "checkpoints/unet"

from keras_segmentation.models.unet import unet
model = unet(n_classes=3, input_height=input_height, input_width=input_width)


if LOAD:
    latest_weights = find_latest_checkpoint(ckpath)
    model.load_weights(latest_weights)

if TRAIN:
    model.train( 
        train_images =  "./train_frames/",
        train_annotations = "./train_masks/",
        checkpoints_path = ckpath , epochs=epochs,
        batch_size=16
    )
    print('FINISHED TRAINING')

##out = model.predict_segmentation(
##    inp="./val_frames/ms_8ply_0000_0_10.png",
##    out_fname="test_output.png"
##)

inp_dir="./12ply/"
out_dir="./outputs/"
regex = inp_dir + "adept12ply_raw_????_?_?.png"
imgpaths = sorted(glob(regex))

### Apply network to target imgs
if APPLY:
    for p in imgpaths:
        folder,name,ext = splitfn(p)
        out = model.predict_segmentation(
            inp=p,
            out_fname=out_dir+name+ext,
            colors=[(0,0,255),(0,255,0),(255,0,0)]
        )
        lname = name.split('_')
        if lname[-1]=='0' and lname[-2]=='0':
            print(name)

##predict_multiple(
##        model=model,
##	checkpoints_path=ckpath, 
##	inp_dir="./12ply/", 
##	out_dir="outputs/",
##        colors=[(0,0,255),(0,255,0),(255,0,0)]
##)
