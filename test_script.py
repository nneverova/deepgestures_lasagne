#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#

import os
import sys

import glob
import time
import numpy


from motionDetector_lasagne import motionDetector
from audioClassifier_lasagne import audioClassifier
from skeletonClassifier_lasagne import skeletonClassifier
from videoFeatureExtractor_lasagne import videoFeatureExtractor
from videoClassifier_lasagne import videoClassifier
from multimodalClassifier_lasagne import multimodalClassifier
from postprocessing import aggregate_predictions, eval_gesture


## Example of a test script

# Modify these to point to the respective paths

input_folder = '/mnt/data/dramacha/data_preprocessedv2/realtest/'
filter_folder = './filters/'
label_folder = './ground_truth/'
prediction_folder = './predictions/'

# choose classifer. passed as argument in cmd line
cl_methods = {'skeleton' : skeletonClassifier,
			  'video' : videoClassifier,
			  'audio' : audioClassifier,
			  'multimodal' : multimodalClassifier
			  }

if len(sys.argv) < 2:
	print "Usage: {0} classifier step".format(sys.argv[0])
	exit(1)
else:
	cl_mode = sys.argv[1]
	step = int(sys.argv[2])

if not cl_mode in cl_methods.keys():
	print 'Unknown classifier. Options:', cl_methods.keys()
	exit(1)

if not step in [2, 3, 4]:
	print '"Step" can take the following values: [ 2 | 3 | 4 ]'
	exit(1)


old_predictions = glob.glob(prediction_folder + '*.csv')

#remove old predictions
for f in old_predictions:
	os.remove(f)

#set appropriate input folders and sample list
if cl_mode == 'video':
	source_folder = input_folder + "/color/"
	#load file list
	samples_list = glob.glob(source_folder + '*.pickle')
	list.sort(samples_list)
	samples_list = samples_list[1::2]  # get unique seqIDs - odd values
	use_video = True
elif cl_mode == 'skeleton':
	source_folder = input_folder + "/mocap/"
	samples_list = glob.glob(source_folder + '*.pickle')
	list.sort(samples_list)
	use_video = False
elif cl_mode == 'audio':
	source_folder = input_folder + "/audio/"
	samples_list = glob.glob(source_folder + '*.pickle')
	list.sort(samples_list)
	use_video = False
elif cl_mode == 'multimodal':
	source_folder = input_folder + "/color/"
	#load file list
	samples_list = glob.glob(source_folder + '*.pickle')
	list.sort(samples_list)
	samples_list = samples_list[1::2]  # get unique seqIDs - odd values
	use_video = True
else:
	sys.exit("Classifier undefined")


# Define and initialize the main classifier
classifier = cl_methods[cl_mode](step = step,
								input_folder = source_folder,
								filter_folder = filter_folder)


classifier.build_network() # build the model
classifier.load_model() # load a pretrained model

# Define and initialize a binary motion detector
motion_detector_lasagne = motionDetector(input_folder = source_folder,
								 filter_folder = filter_folder)
motion_detector_lasagne.build_network()
motion_detector_lasagne.set_filters()


n_samples = len(samples_list)

running_ind = 0

for sample_file in samples_list:

	print 'Processing sample:', running_ind, '/', n_samples
	# get seqID from filename :
	path,filename = os.path.split(sample_file)
	strs = filename.split('_',1)
	seqID = strs[0]
	print "seqID is: %s" % seqID
	#test_sample = classifier.load_sample(sample_file) # load data for the main classifier
	test_sample_motion = motion_detector_lasagne.load_sample(sample_file)

	probs = classifier.test(sample_file) # load data for motion detector
	probs_motion = motion_detector_lasagne.test(test_sample_motion)

	# Aggregate per-block predictions
	aggregate_predictions([[probs, step]], [probs_motion, 1], prediction_folder, seqID)

	running_ind +=1

# Get scores
scb = eval_gesture(prediction_folder,label_folder)
print '\nFinal score %f.'  % (scb)

