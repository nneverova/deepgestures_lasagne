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
import numpy
import random
import re
import math
from scipy.misc import toimage

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import batch_norm, dnn, SliceLayer, FlattenLayer

from skeletonClassifier_lasagne import skeletonClassifier
from videoClassifier_lasagne import videoClassifier
from audioClassifier_lasagne import audioClassifier
from basicClassifier_lasagne import basicClassifier


class multimodalClassifier(basicClassifier):

	def __init__(self, input_folder, filter_folder,
				 number_of_classes=21, step=4, nframes=5, batch_size=42,
				 modality_list=['color','depth','mocap','audio'],
				 pretrained=False, pretrained_paths=False, set_shared_layers=True,
				 do_lmdb = False):

		basicClassifier.__init__(self, input_folder, filter_folder, number_of_classes, step, nframes, batch_size, 'mocap', pretrained, do_lmdb)

		self.modality_list = modality_list
		self.hand_list['right'] = ['color','depth']
		self.hand_list['left'] = ['color','depth']
		self.hand_list['both'] = ['mocap','audio']

		print "Pre-trained paths set to FALSE! End to end training!"


		self.pretrained_paths = pretrained_paths
		self.set_shared_layers = set_shared_layers
		self.nbands = 4

		self.filters_file = filter_folder + 'multimodalClassifier_step' \
			+ str(step) + '.npz'
		self.fc_layers = [self.nbands*self.nclasses, self.nclasses]
		self.dropout_rates = [0., 0.]
		self.activations = [self.activation] * (len(self.fc_layers) - 1)

		lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results

		# Training parameters
		self.learning_rate_value=0.005
		self.learning_rate_decay=0.9998
		self.n_epochs=5000
		self.network = {}
		self.classifiers={}
		self.sinputs = []


	def _load_file(self, file_name, data_sample=None):
		""" FOR PICKLES ONLY """
		data_sample = {}
		#print file_name
		for cl in self.classifiers.values():
		  data_sample = cl._load_file(file_name, data_sample)
		return data_sample

	def _get_stblock(self, data_input, hnd, mdlt, start_frame=None):

		if start_frame is None:
		   start_frame = random.randint(0, len(data_input)
				- self.step * (self.nframes - 1) - 1)
		if not mdlt == 'depth':
		   stblock = self.classifiers[mdlt]._get_stblock(data_input,
				hnd, mdlt, start_frame)
		else:
		   stblock = self.classifiers['color']._get_stblock(data_input,
				hnd, mdlt, start_frame)
		return stblock

	def build_network(self, input_var = None, batch_size = None):
		""" Function to initialize network parameters and build a
			fully connected fusion-layer using Theano

		"""


		print "build network() in multimodalClassifier.py invoked"
		if 'color' in self.modality_list:
		  tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
		  self.sinputs = [tensor5('color_right'), tensor5('depth_right'),
		  	tensor5('color_left'), tensor5('depth_left')]



		self.classifiers['color'] = videoClassifier(input_folder=self.input_folder,
											  filter_folder=self.filter_folder,
											  batch_size=self.batch_size)
		self.network['color_right'] = self.classifiers['color'].right_module.build_network(self.sinputs[:2])
		self.network['color_left'] = self.classifiers['color'].left_module.build_network(self.sinputs[2:4])
		self.network['color'] = self.classifiers['color'].build_network(self.sinputs)


		for mdlt in ['mocap','audio']:

			if mdlt in self.modality_list:
				tensor4 = T.TensorType(theano.config.floatX, (False,) * 4)
				self.sinputs.append(T.tensor4(mdlt))

			if mdlt=='mocap':
				self.classifiers[mdlt] = skeletonClassifier(
								  input_folder=self.input_folder,
													filter_folder=self.filter_folder)
				self.network[mdlt] = self.classifiers[mdlt].build_network(self.sinputs[-1:])
			elif mdlt=='audio':
				self.classifiers[mdlt] = audioClassifier(
								  input_folder=self.input_folder,
													filter_folder=self.filter_folder)
				self.network[mdlt] = self.classifiers[mdlt].build_network(self.sinputs[-1:])


		# concatenate the mocap and audio modalities
		#self.network['ConCatLayer_moc_aud'] = lasagne.layers.ConcatLayer([self.network['mocap']['FC_3'], self.network['audio']['FC_2']], axis=1, cropping=None)


		# concatenate all modalities
		self.network['ConcatLayer_multi'] = lasagne.layers.ConcatLayer([self.network['color']['FC_3'],
		                                      self.network['mocap']['FC_3'], self.network['audio']['FC_2']], axis=1, cropping=None)


		self.network['FC_1'] = batch_norm(lasagne.layers.DenseLayer(
							lasagne.layers.dropout(self.network['ConcatLayer_multi'] , p=self.dropout_rates[0]),
							num_units=120,
							nonlinearity=lasagne.nonlinearities.tanh))

		self.network['FC_2'] = batch_norm(lasagne.layers.DenseLayer(
							lasagne.layers.dropout(self.network['FC_1'] , p=self.dropout_rates[1]),
							num_units=60,
							nonlinearity=lasagne.nonlinearities.tanh))

		self.network['prob'] = batch_norm(lasagne.layers.DenseLayer(
							lasagne.layers.dropout(self.network['FC_2'], p=self.dropout_rates[1]),
							num_units=self.nclasses,
							nonlinearity=lasagne.nonlinearities.softmax))

		self.input_size = {}
		for mdlt in self.modality_list:
		  if not mdlt=='depth':
			self.input_size[mdlt] = self.classifiers[mdlt].input_size[mdlt]
		self.input_size['depth'] = self.input_size['color']


		if self.pretrained_paths:

				filters_pretrained = self.filter_folder + 'videoClassifier_step' + str(self.step) + '.npz'
				with numpy.load(filters_pretrained) as f:
					param_values = [f['arr_%d' % i] for i in range(len(f.files))]
					lasagne.layers.set_all_param_values(self.network['color']['prob'], param_values)
					print "Pre-trained weights for VideoClassifier loaded"


				filters_pretrained = self.filter_folder + 'audioClassifier_step' + str(self.step) + '.npz'
				with numpy.load(filters_pretrained) as f:
					param_values = [f['arr_%d' % i] for i in range(len(f.files))]
					lasagne.layers.set_all_param_values(self.network['audio']['prob'], param_values)
					print "Pre-trained weights for audioClassifier loaded"


				filters_pretrained = self.filter_folder + 'skeletonClassifier_step' + str(self.step) + '.npz'
				with numpy.load(filters_pretrained) as f:
					param_values = [f['arr_%d' % i] for i in range(len(f.files))]
					lasagne.layers.set_all_param_values(self.network['mocap']['prob'], param_values)
					print "Pre-trained weights for skeletonClassifier loaded"


		return self.network





	def train_lasagne(self, learning_rate_value=None, learning_rate_decay=None, num_epochs=2000):


		super(multimodalClassifier,self).train_lasagne(learning_rate_value,
			learning_rate_decay, num_epochs)

