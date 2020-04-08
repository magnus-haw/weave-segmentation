from keras.models import Input,load_model
from keras.layers import Dropout,concatenate,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D

from keras_segmentation.predict import predict_multiple
from keras_segmentation.train import find_latest_checkpoint

TRAIN = True
LOAD = False
input_height,input_width = 64,64
n_classes = 3
epochs = 5

from keras_segmentation.models.unet import unet
model = unet(n_classes=3 ,  input_height=64, input_width=64  )

ckpath = "checkpoints/unet"

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

if not TRAIN:
    latest_weights = find_latest_checkpoint(ckpath)
    model.load_weights(latest_weights)

##predict_multiple(
##        model=model,
##	checkpoints_path=ckpath, 
##	inp_dir="./val_frames/", 
##	out_dir="outputs/",
##        colors=[(0,0,255),(0,255,0),(255,0,0)]
##)
