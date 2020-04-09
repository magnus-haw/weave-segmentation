# weave-segmentation
segmentation of woven materials

## Python environment:
Python 3.7 using openCV 4.2.0, Keras 2.3.1, numpy 1.18, and TensorFlow 2.1
Full conda environment defined in conda-env.txt. To clone the environment use: 
```
conda create --name cnn --file conda-env.txt
```
Must also install keras-segmentation package (`pip install keras-segmentation`)

## Steps to run pre-trained CNN in segmentation folder:
0) Place 12ply 3D tif (adept12ply_raw.tif) in ADEPT folder 
1) Run 12ply_analyze.py with MAKE_TEST_SET=True, REASSEMBLE_OUTPUT = False
2) Run cnn_unet_classifier.py with TRAIN=False, LOAD=True, APPLY=True (takes ~15 min)
3) Run 12ply_analyze.py with MAKE_TEST_SET=False, REASSEMBLE_OUTPUT = True

-outputs placed in 12ply_segmented/ and adept12ply_labels.tiff

## Training CNN
0) Place 4, 6, 8ply 3D tiffs & labeled tiffs in ADEPT folder (adept4ply_raw_05.tif, adept4ply_tagged_05.tif, etc.)
1) Run make_training_set.py
2) Run cnn_unet_classifier.py with TRAIN=True, LOAD=False, APPLY=False (takes ~15 min)
