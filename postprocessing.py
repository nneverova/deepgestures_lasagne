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
import numpy
import math
import csv
from functools import partial

def list_duplicates_of(seq,item):
	start_at = -1
	locs = []
	while True:
		try:
			loc = seq.index(item, start_at + 1)
		except ValueError:
			break
		else:
			locs.append(loc)
			start_at = loc
	return locs

def correct_pos(pos, mt_switch, t_value, max_shift):
	nfrms = len(mt_switch)
	found = False
	shift = 0
	while not found and shift < max_shift:
		if mt_switch[max(0, pos - shift)] == t_value:
			found = True
			pos -= shift
		elif mt_switch[min(pos+shift, nfrms-1)] == t_value:
			found = True
			pos += shift
		elif pos-shift <= 0 and pos+shift >= nfrms-1:
			found = True
		shift +=1
	return pos

def check_gesture_length(gl, min_gl, max_gl):
	if gl < min_gl or gl > max_gl:
		return False
	else:
		return True


def aggregate_predictions(input, motiondata, output_folder, seqID, nframes=5, nclasses=21):

		num_frames = input[0][0].shape[0] + (nframes*input[0][1] + 1) - 1

		num_el = numpy.zeros((num_frames, nclasses))
		preds_acc = numpy.zeros((num_frames,))
		probs_acc = numpy.zeros((num_frames, nclasses))

		preds_acc_m = numpy.zeros((num_frames, ))
		probs_acc_m = numpy.zeros((num_frames, 2))

		for modality in xrange(len(input)):
			window_size = nframes*input[modality][1] + 1
			for i in xrange(input[modality][0].shape[0]):
				probs_acc[i : i+window_size] += input[modality][0][i]
				num_el[i : i+window_size, :] +=1

		probs_acc = numpy.divide(probs_acc, num_el)

		for i in xrange(num_frames):
			preds_acc[i] = numpy.argmax(probs_acc[i],axis=0)


		window_size = nframes*motiondata[1] + 1
		for i in xrange(motiondata[0].shape[0]):
			probs_acc_m[i : i+window_size] += motiondata[0][i]

		for i in xrange(num_frames):
			preds_acc_m[i] = numpy.argmax(probs_acc_m[i],axis=0)

		previous_pred = 0
		motion_switch = numpy.zeros((num_frames,))
		for i in xrange(num_frames):
			if preds_acc_m[i] == 1 and previous_pred == 0:
				motion_switch[i] = 1
			elif preds_acc_m[i] == 0 and previous_pred == 1:
				motion_switch[i] = -1
			previous_pred = preds_acc_m[i]


		return create_sequence(preds_acc, probs_acc, motion_switch, output_folder, seqID, nclasses)


def create_sequence(preds, probs, mswitch, output_folder, seq_ID, nclasses):

	print 'creating a sequence'
	num_frames = len(preds)
	max_shift = 25
	min_gesture_length = 20
	max_gesture_length = 100
	crop_beginning = 4
	crop_ending = 3

	max_score = []
	action_list = []
	start_frame = []
	end_frame = []
	pred_back = 0
	predd = []
	preddd = []

	adjust_range = True

	for i in xrange(num_frames):

		if preds[i] > 0 and not preds[i] == pred_back:
			start_frame.append(i)
			end_frame.append(i)
			action_list.append(int(preds[i]))
			max_score.append(probs[i][preds[i]])
			pred_back = preds[i]

		elif preds[i] > 0:
			max_score[-1] = max(max_score[-1], probs[i][preds[i]])
			end_frame[-1] = i

		elif i < num_frames - 2 and i > 0:
			if preds[i-1] > 0 and preds[i-1] == numpy.any(preds[i + 1 : i + 3]):
				end_frame[-1] = i

	predd = zip(action_list, start_frame, end_frame)

	dups_in_source = partial(list_duplicates_of, action_list)

	for x in xrange(nclasses):
		mm = []
		if len(dups_in_source(x)) > 0:

			for j in dups_in_source(x):
				mm.append(max_score[j])

			pr_to_append = predd[dups_in_source(x)[numpy.argmax(mm)]]

			start_pos = pr_to_append[1]
			end_pos = pr_to_append[2]

			if adjust_range:
				gesture_length = end_pos - start_pos

				start_pos = correct_pos(start_pos, mswitch, 1, max_shift)
				end_pos = correct_pos(end_pos, mswitch, -1, max_shift)
				upd_gesture_length = end_pos - start_pos

				if check_gesture_length(upd_gesture_length, min_gesture_length, max_gesture_length) or not check_gesture_length(gesture_length, min_gesture_length, max_gesture_length):
						pr_to_append = (x, start_pos, end_pos)

				pr_to_append = (x, max(pr_to_append[1], crop_beginning), min(pr_to_append[2], num_frames - crop_ending))

			preddd.append(pr_to_append)
			for i in xrange(pr_to_append[1], pr_to_append[2]):
				mswitch[i] = 10
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	output_filename = os.path.join(output_folder,  'Sample' + seq_ID + '_prediction.csv')
	output_file = open(output_filename, 'wb')
	for row in preddd:
		output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
	output_file.close()
	return preddd

# Script provided by organizers of Chalearn challenge on gesture recognition (2014)
def eval_gesture(prediction_dir, truth_dir):
	""" Perform the overlap evaluation for a set of samples """
	worseVal = 10000

	# Get the list of samples from ground truth
	gold_list = sorted(os.listdir(prediction_dir))
	print gold_list

	# For each sample on the GT, search the given prediction
	numSamples = 0.0;
	score = 0.0;
	for gold in gold_list:
		# Avoid double check, use only labels file
		if not gold.lower().endswith("_prediction.csv"):
			continue

		# Build paths for prediction and ground truth files
		sampleID = gold[0:10] #re.sub('\_labels.csv$', '', gold)
		labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
		dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
		predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")

		# Get the number of frames for this sample
		with open(dataFile, 'rb') as csvfile:
			filereader = csv.reader(csvfile, delimiter=',')
			for row in filereader:
				numFrames = int(row[0])
			del filereader

		if os.path.isfile(predFile):
			# Get the score
			numSamples += 1
			print gesture_overlap_csv(labelsFile, predFile, numFrames)
			score += gesture_overlap_csv(labelsFile, predFile, numFrames)

	return score/numSamples


# Script provided by organizers of Chalearn challenge on gesture recognition (2014)
def gesture_overlap_csv(csvpathgt, csvpathpred, seqlenght):
	""" Evaluate this sample against the ground truth file """
	maxGestures = 20

	# Get the list of gestures from the ground truth and frame activation
	gtGestures = []
	binvec_gt = numpy.zeros((maxGestures, seqlenght))
	with open(csvpathgt, 'rb') as csvfilegt:
		csvgt = csv.reader(csvfilegt)
		for row in csvgt:
			binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
			gtGestures.append(int(row[0]))

	# Get the list of gestures from prediction and frame activation
	predGestures = []
	binvec_pred = numpy.zeros((maxGestures, seqlenght))
	with open(csvpathpred, 'rb') as csvfilepred:
		csvpred = csv.reader(csvfilepred)
		for row in csvpred:
			binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
			predGestures.append(int(row[0]))

	# Get the list of gestures without repetitions for ground truth and predicton
	gtGestures = numpy.unique(gtGestures)
	predGestures = numpy.unique(predGestures)

	bgt = (numpy.argmax(binvec_gt,axis=0)+1) * (numpy.max(binvec_gt,axis=0)>0)
	bpred = (numpy.argmax(binvec_pred,axis=0)+1) * (numpy.max(binvec_pred,axis=0)>0)

	# Find false positives
	falsePos=numpy.setdiff1d(gtGestures,numpy.union1d(gtGestures,numpy.union1d(gtGestures,predGestures)))

	# Get overlaps for each gesture
	overlaps = []
	for idx in gtGestures:
		intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
		aux = binvec_gt[idx-1] + binvec_pred[idx-1]
		union = sum(aux > 0)
		overlaps.append(intersec/union)

	# Use real gestures and false positive gestures to calculate the final score
	return sum(overlaps)/(len(overlaps)+len(falsePos))
