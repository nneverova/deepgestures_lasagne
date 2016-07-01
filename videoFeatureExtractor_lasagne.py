
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
import numpy.ma
import math
import re
import cPickle
import glob

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import batch_norm, dnn, SliceLayer, FlattenLayer


from basicClassifier_lasagne import basicClassifier

class videoFeatureExtractor(basicClassifier):
	def __init__(self, input_folder, filter_folder, number_of_classes=21,
	step=4, nframes=5, block_size=36, batch_size=42, pretrained=False,do_lmdb = False):
		basicClassifier.__init__(self, input_folder, filter_folder, number_of_classes,
									 step, nframes, batch_size, 'color', pretrained, do_lmdb)

			# input parameters
		self.block_size = block_size  # size of a bounding box surrounding each hand
		self.input_size['color'] = [self.nframes, self.block_size, self.block_size]
		self.input_size['depth'] = self.input_size['color']
		self.conv_layers = [(25,3,1,5,5),(25,25,5,5)]
		self.pooling = [(2,2,3),(1,1)]
		self.fc_layers = [900, 450, self.nclasses]
		self.dropout_rates = [0., 0., 0.] # dropout rates for fully connected layers
		self.activations = [self.activation] * (len(self.conv_layers)
										+ len(self.fc_layers) - 1)

		self.modality_list = ['color','depth']
		self.hand_list['both'] = self.modality_list
		self.number_of_classes = 21

		lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results

		# Theano inputs

		tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
		self.sinputs = [tensor5(mdlt) for mdlt in self.modality_list]
		self.network = {}


		# Paths
		self.filters_file = filter_folder + 'videoFeatureExtractor_step' + str(step) + '.npz'

		# Training parameters
		self.learning_rate_value = 0.01
		self.learning_rate_decay = 0.9998
		self.n_epochs = 5000


	def prenormalize(self,x):
		x = x - numpy.mean(x)
		xstd = numpy.std(x)
		return x / (xstd + 0.00001)


	def _get_data_list(self, subset):

		if subset == 'train': folder = self.train_folder
		elif subset == 'valid': folder = self.valid_folder
		elif subset == 'test': folder = self.test_folder
		else: print 'Unknown subset'

		self.data_list[subset] = {}
		for cl in xrange(self.nclasses):
				list_right = glob.glob(folder + "*r%02d*.pickle" % (cl))
				list_left = glob.glob(folder + "*l%02d.pickle" % (cl))
				self.data_list[subset][cl] = list_right + list_left

	def _get_stblock(self, data_input, hnd, mdlt, start_frame=None):
		goodness = False
		if start_frame is None:
				start_frame = random.randint(0, len(data_input['min_length'])-self.step*(self.nframes-1)-1)
		stblock = numpy.zeros([self.nframes, self.block_size, self.block_size])
		for ii in xrange(self.nframes):
				v = data_input[hnd][mdlt][start_frame + ii * self.step]
				mm = abs(numpy.ma.maximum(v))
				if mm > 0.:
						# normalize to zero mean, unit variance,
						# concatenate in spatio-temporal blocks
						stblock[ii] = self.prenormalize(v)
						goodness = True
		return stblock, goodness

	def _load_file(self, file_name, data_sample=None):
		if data_sample is None:
				data_sample = {}
		for hnd in self.hand_list:
				data_sample[hnd] = {}
				for mdlt in self.modality_list:
						if not hnd=='both':
								for ind in ['a','l','r']:
										file_name = re.sub('_'+ind+'_','_'+hnd[0]+'_',file_name)
						for mdl in ['color','depth','mocap','descr','audio']:
								file_name = re.sub(mdl,mdlt,file_name)
						with open(file_name,'rb') as f:
								[data_sample[hnd][mdlt]] = cPickle.load(f)
						if not 'min_length' in data_sample:
								data_sample['min_length'] = len(data_sample[hnd][mdlt])
						else:
								data_sample['min_length'] = min(data_sample['min_length'],len(data_sample[hnd][mdlt]))
		return data_sample



	def build_network(self, input_var=None, batch_size = None):


		if not input_var is None: self.sinputs = input_var
		if not batch_size is None:
			self.batch_size = batch_size



		# Input layer, as usual:
		self.network['Input_color'] = lasagne.layers.InputLayer(shape=(self.batch_size,1,self.block_size,self.block_size,self.nframes),
												  input_var=self.sinputs[0].dimshuffle(0,2,3,4,1))
		# 3D convolutional layer
		self.network['Conv3D_color'] = batch_norm(lasagne.layers.dnn.Conv3DDNNLayer(self.network['Input_color'], num_filters=25, filter_size=(5,5,3), stride=1,
																	 pad='valid', untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
																	 nonlinearity=lasagne.nonlinearities.tanh, flip_filters=False))
		# 3D Max-pooling of 3D conv layer
		self.network['MaxPool3D_color'] = lasagne.layers.dnn.Pool3DDNNLayer(self.network['Conv3D_color'], pool_size=(2,2,3), stride=None, pad=(0, 0, 0), ignore_border=True, mode='max')


		#re-shape dimensions of output into a 4-dimensional input to the 2D-convolutional layer next
		self.network['Flatten_Layer_1'] = lasagne.layers.FlattenLayer(self.network['MaxPool3D_color'],outdim=4)

		# 2D convolutional layer
		self.network['Conv2D_color'] = batch_norm(lasagne.layers.dnn.Conv2DDNNLayer(self.network['Flatten_Layer_1'], 25, filter_size=(5,5), stride=1, pad=0, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.tanh, flip_filters=False))

		# Max-pooling layer of factor 2 in both dimensions:
		self.network['MaxPool2D_color'] = lasagne.layers.dnn.MaxPool2DDNNLayer(self.network['Conv2D_color'], pool_size=(1,1), stride=None, pad=(0, 0), ignore_border=True)

		self.network['Flatten_Layer_2'] = lasagne.layers.FlattenLayer(self.network['MaxPool2D_color'], outdim=2)

		## Input layer, as usual:
		self.network['Input_depth'] = lasagne.layers.InputLayer(shape=(self.batch_size,1,self.block_size,self.block_size,self.nframes),
												  input_var=self.sinputs[1].dimshuffle(0,2,3,4,1))
		# 3D convolutional layer
		self.network['Conv3D_depth'] = batch_norm(lasagne.layers.dnn.Conv3DDNNLayer(self.network['Input_depth'], num_filters=25, filter_size=(5,5,3), stride=1,
																	 pad='valid', untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
																	 nonlinearity=lasagne.nonlinearities.tanh, flip_filters=False))
		# 3D Max-pooling of 3D conv layer
		self.network['MaxPool3D_depth'] = lasagne.layers.dnn.Pool3DDNNLayer(self.network['Conv3D_depth'], pool_size=(2,2,3), stride=None, pad=(0, 0, 0), ignore_border=True, mode='max')

		# re-shape dimensions of output into a 4-dimensional input to the 2D-convolutional layer next
		self.network['Flatten_Layer_3'] = lasagne.layers.FlattenLayer(self.network['MaxPool3D_depth'],outdim=4)

		#2D convolutional layer
		self.network['Conv2D_depth']  = batch_norm(lasagne.layers.dnn.Conv2DDNNLayer(self.network['Flatten_Layer_3'], 25, filter_size=(5,5), stride=1, pad=0, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.tanh, flip_filters=False))

		# Max-pooling layer of factor 2 in both dimensions
		self.network['MaxPool2D_depth'] = lasagne.layers.dnn.MaxPool2DDNNLayer(self.network['Conv2D_depth'], pool_size=(1,1), stride=None, pad=(0, 0), ignore_border=True)

		self.network['Flatten_Layer_4'] = lasagne.layers.FlattenLayer(self.network['MaxPool2D_depth'], outdim=2)

		##merge both depth and color layers
		self.network['Fusion_layer'] = lasagne.layers.ConcatLayer([self.network['Flatten_Layer_2'],self.network['Flatten_Layer_4']], axis=1, cropping=None)

		#Fully connected Layer #1
		self.network['FC_1'] = batch_norm(lasagne.layers.DenseLayer( lasagne.layers.dropout(self.network['Fusion_layer'], p=.0),
													 num_units=900,
													 nonlinearity=lasagne.nonlinearities.tanh))
		#Fully connected Layer #2
		self.network['FC_2'] = batch_norm(lasagne.layers.DenseLayer( lasagne.layers.dropout(self.network['FC_1'] , p=.0),
													 num_units=450,
													 nonlinearity=lasagne.nonlinearities.tanh))


		# Output Layer
		self.network['prob'] = batch_norm(lasagne.layers.DenseLayer(
				lasagne.layers.dropout(self.network['FC_2'], p=.0),
				num_units=self.number_of_classes,
				nonlinearity=lasagne.nonlinearities.softmax))


		return self.network









