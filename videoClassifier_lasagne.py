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


__docformat__ = 'restructedtext en'

import os
import sys
import time
import datetime
import numpy
import scipy
import scipy.io
import collections
import random
import glob
import numpy.ma
import math
import re

import theano
from theano import tensor as T, function, printing


from basicClassifier_lasagne import basicClassifier
from videoFeatureExtractor_lasagne import videoFeatureExtractor

import lasagne
from lasagne.layers import batch_norm, dnn, SliceLayer, FlattenLayer



class videoClassifier(videoFeatureExtractor):

  """ Gesture recognition based on RGB and depth data (from hand images)
  """

  def __init__(self, input_folder, filter_folder, number_of_classes=21, step=4, nframes=5,
		block_size=36, batch_size=42, use_standard_features=True, pretrained=False,
		do_lmdb=False):


	videoFeatureExtractor.__init__(self, input_folder, filter_folder,
			number_of_classes, step, nframes, block_size, batch_size,
			pretrained, do_lmdb)

	lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results

	# network parameters
	self.fc_layers = [900, 2*self.nclasses, self.nclasses]
	self.dropout_rates = [0.0, 0.0, 0.0] # dropout rates for fully connected layers
	self.activations = [self.activation] * (len(self.fc_layers) - 1)

	# symbolic inputs
	self.hand_list = {}
	self.network = {}
	self.sinputs = []
	for hnd in ['right','left']:
	  self.hand_list[hnd] = self.modality_list
	tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
	for hnd in self.hand_list:
	  for mdlt in self.hand_list[hnd]:
		  self.sinputs.append(tensor5(mdlt+'_'+hnd))

	# training parameters
	self.learning_rate_value = 0.02
	self.learning_rate_decay = 0.998
	self.n_epochs = 5000

	# paths
	self.filters_file   = filter_folder + \
			'videoClassifier_step' + str(step) + '.npz'
	self.search_line = "*_r_color_g%02d*.pickle"


		# Load feature extractors (same for images of right and left hands)
	self.right_module = videoFeatureExtractor(input_folder=self.input_folder, filter_folder=self.filter_folder)
	self.right_network = self.right_module.build_network(input_var=self.sinputs[:2],batch_size=self.batch_size)
	if use_standard_features:
	  try:
		filters_pretrained = filter_folder + 'videoFeatureExtractor_step' + str(step) + '.npz'
		with numpy.load(filters_pretrained) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			lasagne.layers.set_all_param_values(self.right_network['prob'], param_values)
			print "loaded"
	  except IOError as e:
		print "I/O error({0}): {1}".format(e.errno, e.strerror)
		print 'Pretrained feature extractors not found:', \
					filters_pretrained


	# Load feature extractors (same for images of right and left hands)
	self.left_module = videoFeatureExtractor(input_folder=self.input_folder, filter_folder=self.filter_folder)
	self.left_network = self.left_module.build_network(input_var=self.sinputs[2:],batch_size=self.batch_size)
	if use_standard_features:
	  try:
		filters_pretrained = filter_folder + 'videoFeatureExtractor_step' + str(step) + '.npz'
		with numpy.load(filters_pretrained) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			lasagne.layers.set_all_param_values(self.left_network['prob'], param_values)
	  except IOError as e:
		print "I/O error({0}): {1}".format(e.errno, e.strerror)
		print 'Pretrained feature extractors not found:', \
					filters_pretrained


  def build_network(self, input_var = None, batch_size = None):


		print "build_network in VideoClassifier executed.."
		print "inputs are : " , self.sinputs

		if not input_var is None: self.sinputs = input_var
		if not batch_size is None:
			self.batch_size = batch_size


		# Merge or fuse or concatenate incoming layers
		self.network['ConcatLayer'] = lasagne.layers.ConcatLayer([self.right_network['FC_2'], self.left_network['FC_2']], axis=1, cropping=None)


		self.network['FC_3'] = batch_norm(lasagne.layers.DenseLayer(
							lasagne.layers.dropout(self.network['ConcatLayer'], p=self.dropout_rates[0]),
							num_units=84,
							nonlinearity=lasagne.nonlinearities.tanh))


		self.network['prob'] = batch_norm(lasagne.layers.DenseLayer(
							lasagne.layers.dropout(self.network['FC_3'], p=self.dropout_rates[2]),
							num_units=self.fc_layers[2],
							nonlinearity=lasagne.nonlinearities.softmax))



		return self.network



  def _get_data_list(self, subset):
	""" FOR PICKLES ONLY """
	return basicClassifier._get_data_list(self, subset)


  def prenormalize(self,x):
	  x = x - numpy.mean(x)
	  xstd = numpy.std(x)
	  return x / (xstd + 0.00001)



