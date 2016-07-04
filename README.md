## DeepGestures ##

Original Theano implementation by Natalia Neverova
natalia.neverova@gmail.com

Re-implementation of DeepGestures using Lasagne by Dhanesh Ramachandram
dhaneshr@gmail.com

The code was tested with Theano 0.82 and Lasagne 0.2dev1 and Anaconda Python
on Ubuntu 14.04 running CuDNN version 4.0


## Prerequisites ##

1. Bleeding-edge version of Theano
   Install using : 
```
#!python
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

```

2. Bleeding edge version of Lasagne
   Install using :


```
#!python
pip install --upgrade --no-deps https://github.com/Lasagne/Lasagne/archive/master.zip

```

3. CuDNN 
Download CuDNN v4 from NVIDIA website. Extract the library into a folder on your home directory. 
Then add the following lines in ~/.bash_profile



```
#!bash

export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/path/to/cuda/include:$CPATH
export LIBRARY_PATH=/path/to/cuda:$LIBRARY_PATH

```



## How to ##

1. training_script.py is the main script to initiate training. In this file, modify the path to the Montalbano dataset accordingly.
It should point toward a preprocessed version of the Montalbano dataset which can be found at this Google Drive Link (note: it is 35 GB):  
https://drive.google.com/open?id=0B2A1tnmq5zQdZEpBX2REcXdpaUU  
We are working on cleaning up and releasing a set of scripts to provide the full pre-processing routines to convert the original data to our format. The original data is available at http://sunai.uoc.edu/chalearnLAP/ (Track 3: Gesture recognition).

2. initiate the training by calling 
   
```
#!python

python training_script.py modality step  
```


Replace modality with 'skeleton' for mocap, 'audio' for audio, 'videoFeat' for video,depth  pretraining, 'video' for video-depth fusion/pre training and finally 'multimodal' for all modalities. skeleton and audio can be trained independently, but videoFeat must be trained, before video. 

All other modalities must be pre-trained before multimodal.

'step' can be 2 or 4, but in practice, 4 gives best results.

Also, required for testing phase, a classifier called motionDetector must be called with step = 1 using the training script. 

In the same folder as your source-code, there must be several subfolders created. 

1. subfolder 'filters' - used to store saved models. There is a file called skeleton_stats which must be present in the folder.
2. subfolder 'ground_truth' which contains the ground_truth csv files used for testing. 
