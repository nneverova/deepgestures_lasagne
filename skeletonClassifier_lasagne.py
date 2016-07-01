
"""
 Class that implements gesture classifier on the mocap data.

 Team:        LIRIS (Natalia Neverova, Christian Wolf and Graham Taylor)

 Author:      Natalia Neverova, natalia.neverova@liris.cnrs.fr

 Created:     15/05/2015
 Copyright:   (c) Natalia Neverova

 License:     GNU General Public License

 The programs and documents are distributed without any warranty, express
 or implied.  As the programs were written for research purposes only, they
 have not been tested to the degree that would be advisable in any
 important application.
"""

__docformat__ = 'restructedtext en'


import os
import sys
import datetime
import time
import numpy
import collections
import math
import cPickle
import random
import re
import glob

import theano
import theano.tensor as T


import lasagne
from lasagne.layers import batch_norm

from basicClassifier_lasagne import basicClassifier



class skeletonClassifier(basicClassifier):
  """
  Gesture classification from skeleton

  """

  def __init__(self, input_folder, filter_folder, number_of_classes=21, step=4,
  	nframes=5, batch_size=42, pretrained=False, do_lmdb=False):

	"""
	Initialize input and network parameters

		:type input_folder: string
		:param input_folder: path to folder containing input data

		:type filter_folder: string
		:param filter_folder: path to directory to store/load network weights

		:type number_of_classes: int
		:param number_of_classes: number of classes in input data (default = 21)

		:type step: int
		:param step: no. of temporal 'steps' used for training/test (default = 4)

		:type nframes: int
		:param nframes: number of frames concatenated in a single
			training / test sample (default = 5)


	"""
	basicClassifier.__init__(self, input_folder, filter_folder,
		number_of_classes, step, nframes, batch_size, 'mocap', pretrained, do_lmdb)

	# input parameters
	self.stats_file = filter_folder + 'skeleton_stats'
	if self.do_lmdb:
	  self.dlength = 172 # length of  pose-descriptor
	  self._load_stats_lmdb()
	else:
	  self.dlength = 183
	  self._load_stats()
	self.input_size['mocap'] = [self.nframes, self.dlength]
	self.modality_list = ['mocap']
	self.hand_list['both'] = self.modality_list

	  # network parameters
	self.dropout_rates = [0.0, 0.0, 0.0, 0.0]
	self.fc_layers = [700, 400, 350, self.nclasses]
	self.activations = [self.activation] * (len(self.fc_layers)-1)

	lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results

	# paths
	self.filters_file = filter_folder + 'skeletonClassifier_step' + str(step) + '.npz'

	# training parameters
	self.learning_rate_value = 0.2
	self.learning_rate_decay = 0.995
	self.n_epochs = 10000
	self.network= {}


  def _normalize_descriptor1(self, dscr):
	""" FOR PICKLES ONLY """
	dscr = numpy.subtract(dscr, self.mdt_mean2)
	return numpy.divide(dscr,numpy.sqrt(self.vdt_mean2))

  def _normalize_descriptor0(self, dscr):
	""" FOR PICKLES ONLY """
	dscr = numpy.subtract(dscr, self.mdt_mean1)
	return numpy.divide(dscr,numpy.sqrt(self.vdt_mean1))

  def _normalize(self, dscr):
	return (dscr - self.dmean)/self.dstd

  def _create_descriptor(self, dscr0, dscr1):
   """ FOR PICKLES ONLY """
   dscr0 = self._normalize_descriptor0(dscr0)
   dscr1 = self._normalize_descriptor1(dscr1)
   return numpy.concatenate((dscr0,dscr1),axis=1)

  def _load_file(self, file_name, data_sample=None):
	""" FOR PICKLES ONLY """

	for mdlt in ['/depth/','/color/','/audio/']:
	  file_name = re.sub(mdlt,'/mocap/',file_name)

	for hnd in ['_r_','_l_']:
		file_name = re.sub(hnd,'_a_',file_name)
	for suff in ['depth','color','audio']:
		file_name = re.sub(suff,'descr',file_name)

	if data_sample is None: data_sample = {}

	for hnd in self.hand_list:
	  if not hnd in data_sample:
	  	data_sample[hnd] = {}
	  for mdlt in self.modality_list:
	  	with open(file_name,'rb') as f:
	  	  [d0, d1] = cPickle.load(f)
		data_sample[hnd][mdlt] = self._create_descriptor(d0, d1)
	if not 'min_length' in data_sample:
	  data_sample['min_length'] = len(data_sample[hnd][mdlt])
	else:
	  data_sample['min_length'] = min(data_sample['min_length'],
			len(data_sample[hnd][mdlt]))
	return data_sample

  def _get_stblock(self, data_input, hnd, mdlt, start_frame=None):
  	  """ FOR PICKLES ONLY """
	  if start_frame is None:
		start_frame = random.randint(0, data_input['min_length']
			- self.step * (self.nframes - 1) - 1)
	  stblock = data_input[hnd][mdlt][start_frame : start_frame
	  		+ self.step * (self.nframes - 1) + 1 : self.step]
	  return stblock, True

  def _load_stats_lmdb(self):
	""" FOR LMDB ONLY

		Function to load normalized data (zero-mean & variance)
		"""
	f = file('filters/skeleton_stats_173','r')
	self.dmean = cPickle.load(f)
	self.dstd = cPickle.load(f)
	f.close()

  def _load_stats(self):
	""" FOR PICKLES ONLY """
	f = open(self.stats_file,'r')
	self.mdt_mean1 = cPickle.load(f)
	self.mdt_mean2 = cPickle.load(f)
	self.vdt_mean1 = cPickle.load(f)
	self.vdt_mean2 = cPickle.load(f)
	f.close()


	# Theano variables
	tensor4 = T.TensorType(theano.config.floatX, (False,) * 4)
	self.sinputs = [T.tensor4('x')]


  def build_network(self, input_var=None, batch_size = None):

		print "build_network() in SkeletonClassifier invoked"
		print self.sinputs

		if not input_var is None: self.sinputs = input_var
		if batch_size: self.batch_size = batch_size



		if not input_var is None: self.sinputs = input_var
		if not batch_size is None:
			self.batch_size = batch_size

		self.network['input'] = lasagne.layers.InputLayer(shape=(self.batch_size,self.nframes,1,self.dlength), input_var=self.sinputs[0])

		self.network['FC_1'] = batch_norm(lasagne.layers.DenseLayer( lasagne.layers.dropout(self.network['input'], p=self.dropout_rates[1]),
					num_units=self.fc_layers[0],nonlinearity=lasagne.nonlinearities.tanh))

		self.network['FC_2'] = batch_norm(lasagne.layers.DenseLayer(
					lasagne.layers.dropout(self.network['FC_1'], p=self.dropout_rates[2]),
					num_units=self.fc_layers[1],
					nonlinearity=lasagne.nonlinearities.tanh))

		self.network['FC_3'] = batch_norm(lasagne.layers.DenseLayer(
					lasagne.layers.dropout(self.network['FC_2'], p=self.dropout_rates[3]),
					num_units=self.fc_layers[2],
					nonlinearity=lasagne.nonlinearities.tanh))

		self.network['prob'] = lasagne.layers.DenseLayer(
					lasagne.layers.dropout(self.network['FC_3'], p=.2),
					num_units=self.nclasses,
					nonlinearity=lasagne.nonlinearities.softmax)

		return self.network

