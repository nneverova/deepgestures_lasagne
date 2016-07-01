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
from scipy.misc import toimage
from multiprocessing import Process, Queue

import zmq
import lasagne
import dill as pickle
import theano
import theano.tensor as T
from collections import OrderedDict

import resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6) # for saving theano graphs

current_time=time.strftime("%Y%m%d_%H%M%S")+"_"+str(numpy.random.randint(100000))

class basicClassifier(object):
	def __init__(self, input_folder, filter_folder, number_of_classes=21,
				 step=4, nframes=5, batch_size=42, modality='mocap', pretrained=False,
				 do_lmdb = False):

		"""
		Meta class for all classifiers

		:type input_folder: string
		:param input_folder: path to folder containing input data

			:type filter_folder: string
		:param filter_folder: path to directory to store/load network weights

		:type number_of_classes: int
		:param number_of_classes: number of classes in input data (default = 21)

		:type step: int
		:param step: temporal sampling stride (default = 4)

		:type nframes: int
		:param nframes: number of frames in a spatio-temporal block (default = 5)

		:type batch_size: int
		:param batch_size: batch size for mini-batch training (default = 42)

		:type modality: str
		:param modality: modality defining base path for data Loading

		:type pretrained: bool
		:param pretrained: indicates whether pretrained model is loaded
		"""

		self.signature = current_time

		# Input parameters
		self.nclasses = number_of_classes
		self.step = step
		self.nframes = nframes
		self.seq_per_class = 200
		self.do_lmdb = do_lmdb

		self.modality = modality
		self.hand_list = collections.OrderedDict()
		self.input_size = {}
		self.params=[]

		self.dataset = {}
		self.dataset['train'] = {}
		self.dataset['valid'] = {}
		self.dataset['test'] = {}
		self.data_list = {}

		# Theano variables
		tensor4 = T.TensorType(theano.config.floatX, (False,) * 4)
		tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)


		# Network parameters
		self.conv_layers = []
		self.pooling = []
		self.fc_layers = []
		self.dropout_rates = []
		self.activation = 'relu'
		self.use_bias = True
		self.mask_weights = None
		self.mask_biases = None

		# Training parameters
		lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results
		self.batch_size = 42
		self.pretrained = pretrained
		self.learning_rate_value = 0.05
		self.learning_rate_decay = 0.999
		self.epoch_counter = 1

		# Paths
		self.search_line = "*_g%02d*.pickle"
		self.input_folder = input_folder
		self.train_folder = self.input_folder + modality + '/train/'
		self.valid_folder = self.input_folder + modality + '/valid/'
		self.test_folder = self.input_folder + modality + '/test/'
		self.filter_folder = filter_folder
		self.filters_file = filter_folder + modality + 'Classifier_step' + str(step) + '.npz'


	def load_model(self, filters_pretrained=None):
		"""
		Function to load / set pretrained weights (if it exists) or 'filters'
		:type filters_pretrained: string
		:param filters_pretrained: file that stores pretrained weights(filters)
		"""
		if filters_pretrained is None:
			filters_pretrained = self.filters_file #+ '_pretrained'


		with numpy.load(filters_pretrained) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.network['prob'], param_values)

	def _save_model(self):
		"""
		Function to save all parameters of a model
		"""
		f = open(self.filters_file, 'w')
		numpy.savez(f, lasagne.layers.get_all_param_values(self.network['prob']))
		f.close()

	def _get_data_list(self, subset):
		"""
		!!!WON'T BE NEEDED FOR LMDB DATASETS.

		Function to retrieve list of training data filenames

		:type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
		"""
		if subset == 'train': folder = self.train_folder
		elif subset == 'valid': folder = self.valid_folder
		elif subset == 'test': folder = self.test_folder
		else: print 'Unknown subset'

		self.data_list[subset] = {}
		for cl in xrange(self.nclasses):
			self.data_list[subset][cl] = glob.glob(folder + self.search_line %(cl))




	def _load_dataset(self, subset):
		"""
		!!!WON'T BE NEEDED FOR LMDB DATASETS.

		Function to load Pickle dataset from folder train, validation and test

		:type subset: string
		:param subset: string representing 'train', 'validation' or 'test' subsets
		"""

		# Allocate memory for each modality
		for hnd in self.hand_list:
			self.dataset[subset][hnd] = {}
			for mdlt in self.hand_list[hnd]:
				self.dataset[subset][hnd][mdlt] = numpy.zeros([self.seq_per_class \
															   * self.nclasses] + self.input_size[mdlt])
		self.dataset[subset]['labels'] = numpy.zeros((self.seq_per_class * self.nclasses,))

		sample = 0
		class_number = 0

		#Loading the data
		while sample < self.nclasses * self.seq_per_class:
			# Loading random gesture from a given class
			file_number = random.randint(0, len(self.data_list[subset][class_number]) - 1)
			file_name = self.data_list[subset][class_number][file_number]
			data_sample = self._load_file(file_name)

			# Extract a random spatio-temporal block from the file
			if data_sample['min_length'] >= self.step * (self.nframes - 1) + 1:
				seq_number = random.randint(0, data_sample['min_length'] -
											self.step*(self.nframes - 1) - 1)
				ifloaded = False
				for hnd in self.hand_list:
					for mdlt in self.hand_list[hnd]:
						self.dataset[subset][hnd][mdlt][sample], ifl = \
							self._get_stblock(data_sample,hnd,mdlt,seq_number)
						ifloaded = ifl | ifloaded
				# If the block is loaded, proceed to the next class
				if ifloaded:
					self.dataset[subset]['labels'][sample] = class_number
					sample += 1
					class_number += 1
					if class_number == self.nclasses:
						class_number = 0

		# Reshape the data, convert to floatX
		for hnd in self.hand_list:
			for mdlt in self.hand_list[hnd]:
				if self.input_size[mdlt][0]==self.nframes:
					self.dataset[subset][hnd][mdlt] = \
						self.dataset[subset][hnd][mdlt].reshape([self.seq_per_class \
																 * self.nclasses, self.input_size[mdlt][0], 1] \
																+ self.input_size[mdlt][1:]).astype(theano.config.floatX)
				else:
					self.dataset[subset][hnd][mdlt] = \
						self.dataset[subset][hnd][mdlt].reshape([self.seq_per_class \
																 * self.nclasses,1] \
																+ self.input_size[mdlt]).astype(theano.config.floatX)

		self.dataset[subset]['labels'] = numpy.int8(self.dataset[subset]['labels'])

		#self.dataset[subset][hnd][mdlt] is the 'input'
		#self.dataset[subset]['labels'] is the 'target'

	def _load_sample(self, input_file):
		"""
		!!!WON'T BE NEEDED FOR LMDB DATASETS.

		Prepare a single test sequence
		:type input_file: string
		:param input_file: filename of the test sample
		"""
		data_sample = self._load_file(input_file)
		nseq = data_sample['min_length'] - self.step * (self.nframes - 1)
		print "nseq=" , nseq
		if nseq <= 0:
			return 0, 0
		else:
			nseq_batch = int(numpy.ceil(float(nseq)/float(self.batch_size))*self.batch_size)
			# Allocate memory, make it divisible by batch_size
			for hnd in self.hand_list:
				self.dataset['test'][hnd] = {}
				for mdlt in self.hand_list[hnd]:
					self.dataset['test'][hnd][mdlt] = numpy.zeros([nseq_batch] + self.input_size[mdlt])
		# Extract all spatio-temporal blocks
					for seq_number in range(nseq):
						#try:
						self.dataset['test'][hnd][mdlt][seq_number], _ = \
							self._get_stblock(data_sample,hnd,mdlt,seq_number)
						# except IndexError, e:
						#	  continue

					# Reshape, convert to floatX
					if self.input_size[mdlt][0]==self.nframes:
						self.dataset['test'][hnd][mdlt] = \
							self.dataset['test'][hnd][mdlt].reshape([nseq_batch, self.input_size[mdlt][0], 1] \
																	+ self.input_size[mdlt][1:]).astype(theano.config.floatX)
					else:
						self.dataset['test'][hnd][mdlt] = \
							self.dataset['test'][hnd][mdlt].reshape([nseq_batch, 1] + self.input_size[mdlt]).astype(theano.config.floatX)
			return nseq, nseq_batch / self.batch_size




	def train_lasagne(self, learning_rate_value=None, learning_rate_decay=None, num_epochs=50):
		# Load the dataset

		self.saved_params = []

		y = T.ivector('y')
		learning_rate = T.fscalar('learning_rate')
		epoch = T.fscalar('epoch')


		if self.pretrained:
			print "Saved model found. Loading model..."
			self.load_model()


		if learning_rate_value is None:
			learning_rate_value = self.learning_rate_value
		if learning_rate_decay is None:
			learning_rate_decay = self.learning_rate_decay

		# Create neural network model (depending on first command line parameter)
		print("Building model and compiling functions...")

		print self.sinputs



		# Create a loss expression for training, i.e., a scalar objective we want
		# to minimize (for our multi-class problem, it is the cross-entropy loss):
		prediction = lasagne.layers.get_output(self.network['prob'], deterministic=False)

		loss = lasagne.objectives.categorical_crossentropy(prediction, y)
		loss = loss.mean()
		# We could add some weight decay as well here, see lasagne.regularization.

		# Create update expressions for training, i.e., how to modify the
		# parameters at each training step. Here, we'll use Stochastic Gradient
		# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.

		params = lasagne.layers.get_all_params(self.network['prob'], trainable=True)

		updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate, momentum=0.8)

		#updates = lasagne.updates.adam(loss,params,learning_rate=self.learning_rate_value, beta1=0.9, beta2=0.999, epsilon=1e-08)

		# Create a loss expression for validation/testing. The crucial difference
		# here is that we do a deterministic forward pass through the network,
		# disabling dropout layers.

		test_prediction = lasagne.layers.get_output(self.network['prob'], deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,y)
		test_loss = test_loss.mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
						  dtype=theano.config.floatX)





		# Compile a function performing a training ste on a mini-batch (by giving
		# the updates dictionary) and returning the corresponding training loss:

		#print theano.printing.pprint(loss)
		#theano.printing.pydotprint(loss, outfile="./test2.png", var_with_name_simple=True)
		start_time=time.time()
		train_fn = theano.function(self.sinputs + [y,learning_rate], loss, updates=updates,on_unused_input='ignore')
		print "Train function compilation: "+str(time.time()-start_time)


		start_time=time.time()
		val_fn = theano.function(self.sinputs + [y], [test_loss, test_acc],on_unused_input='ignore')
		print "Validation function compilation: "+str(time.time()-start_time)


		# Loading the training and validation set
		# Loading the validation set
		self._get_data_list('valid')
		self._load_dataset('valid')
		self._get_data_list('train')

		n_train_batches = self.nclasses * self.seq_per_class / self.batch_size
		n_valid_batches = self.nclasses * self.seq_per_class / self.batch_size


		# Finally, launch the training loop.
		print("Starting training...")
		# We iterate over epochs:




		self.best_validation_acc = -numpy.inf
		done_looping = False

		start_time = time.time()
		for epoch in range(num_epochs):
			self._load_dataset('train')

			# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			for minibatch_index in xrange(n_train_batches):
				train_inputs = self._prepare_inputs('train',minibatch_index)
				train_err += train_fn(*train_inputs, learning_rate=learning_rate_value)
				train_batches += 1



			# And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			for minibatch_index in xrange(n_valid_batches):
				valid_inputs = self._prepare_inputs('valid',minibatch_index)
				err, acc = val_fn(*valid_inputs)
				val_err += err
				val_acc += acc
				val_batches += 1

			this_validation_acc = val_acc/val_batches


			if this_validation_acc > self.best_validation_acc:
				self.best_validation_acc = this_validation_acc
				f = open(self.filters_file, 'w')
				numpy.savez(f, *lasagne.layers.get_all_param_values(self.network['prob']))
				f.close()



			learning_rate_value = learning_rate_value * learning_rate_decay

			print(OrderedDict([("epoch", epoch+1), ("time", time.time()-start_time),
			("train_nll", train_err/train_batches), ("val_nll", val_err/val_batches),
			("val_acc", val_acc/val_batches * 100)]))



	def _prepare_inputs(self, subset, ind):
		"""
		Function to sample and concatenate inputs for each minibatches,
		used with pickle files.

		:type subset: str
		:param subset: data subset ("train", "valid" or "test")

		:type ind: int
		:param ind: minibatch index
		"""

		inputs = []

		# Append data from all channels


		for hnd in self.hand_list:
			for mdlt in self.hand_list[hnd]:
				inputs.append(self.dataset[subset][hnd][mdlt][ind * self.batch_size:
															  (ind + 1) * self.batch_size])
		if subset in ['train','valid']:
			inputs.append(self.dataset[subset]['labels'][ind * self.batch_size:
														 (ind + 1) * self.batch_size])
		return inputs
		# this is a list of tuples of size (batch_size, channel=1, input_size)



	def _print_results(self, loss, ts, epoch_counter, learning_rate_value):
		"""
		Function to print out training log

		:type loss: float
		:param loss: loss value

		:type ts: time stamp
		:param ts: time stamp (start time of a given epoch)

		:type epoch_counter: int
		:param epoch_counter: epoch number

		:type learning_rate_value: float
		:param learning_rate_value: current value of the learning rate

		"""
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') # record timestamp
		print "{}, {}, iteration {}, validation error {}, lr={}{}".format(st,
																		  self.filters_file.split('/')[-1],
																		  epoch_counter,
																		  loss * 100,
																		  learning_rate_value,
																		  " **" if loss < self.best_validation_loss else "")

	def test(self, file_name):
		"""
		Function to test trained network on a sample.
		Returns probabilities of the sample belonging to a given class.

		:type file_sample: string
		:param file_sample: filename of the test sample
		"""
		data_length, n_batches = self._load_sample(file_name)
		if data_length<=0:
			print 'Bad sample'
			return
		else:
			get_probabilities = theano.function(self.sinputs, lasagne.layers.get_output(self.network['prob'], deterministic=True), on_unused_input='warn')
			probs = [get_probabilities(*self._prepare_inputs('test',iii)) for iii in xrange(n_batches)]
		return numpy.vstack(probs)[:data_length]



	def test_lasagne(self):
		'''
		Function to obtain test accuracy using data in 'test' folder (not 'real_test') using pre-compiled validation function.
		Implemented by Mike. This is not the function Natalia used in her paper.
		'''
		start_time=time.time()
		if(os.path.exists("theano_val_fn.pkl")):
			f=open("theano_val_fn.pkl","rb")
			test_fn = pickle.load(f)
			f.close()
		else:
			raise ValueError("Couldn't find stored validation function")

		print "Test function loading: "+str(time.time()-start_time)

		self._get_data_list('test')
		self._load_dataset('test')

		test_inputs=[]

		for hnd in self.hand_list:
			for mdlt in self.hand_list[hnd]:
				test_inputs.append(self.dataset["test"][hnd][mdlt][:])
		test_inputs.append(self.dataset["test"]['labels'][:])

		test_err, test_acc = test_fn(*test_inputs)

		print("Test output: ")
		result = OrderedDict([("test_nll", test_err), ("test_acc", test_acc * 100)])
		print result
		return result
