
#  #  This program is free software; you can redistribute it and/or modify
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


"""
This is the implementation of a feedforward MLP that is used to perform
motion detection for teh video

 Team:        LIRIS (Natalia Neverova, Christian Wolf and Graham Taylor)

 Author:      Natalia Neverova, natalia.neverova@liris.cnrs.fr


 Created:     15/05/2015
 Copyright:   (c) Natalia Neverova

 License:     GNU General Public License

 The programs and documents are distributed without any warranty, express
 or implied.  As the programs were written for research purposes only, they
 have not been tested to the degree that would be advisable in any
 important application.
 All use of these programs is entirely at the user's own risk.
"""



__docformat__ = 'restructedtext en'


import os
import sys
import numpy
import collections
import math
import cPickle
import random
import re
import glob
import theano
import theano.tensor as T
import time
import datetime




import lasagne
from lasagne.layers import batch_norm



###################
# BASIC FUNCTIONS #
###################

class motionDetector(object):
	def __init__(self, input_folder, filter_folder, step=1,  nframes=5, do_lmdb = False):

		"""
		Initialize input and network parameters

		:type input_folder: string
		:param input_folder: path to folder containing input data

		:type filter_folder: string
		:param filter_folder: path to directory to store/load network weights

		:type step: int
		:param step: for motionDetector, each frame is processed. (default = 1)

		:type nframes: int
		:param nframes: number of frames used in each scale during training / test (default = 5)


		"""
		lasagne.random.set_rng(numpy.random.RandomState(1234))

		self.nclasses       = 2
		self.step           = step
		self.nframes        = nframes

		self.dlength        = 183


		self.x              = T.matrix('x')
		self.y      		= T.ivector('y')

		self.input_folder   = input_folder
		self.train_folder   = self.input_folder + 'mocap/train/'
		self.valid_folder   = self.input_folder + 'mocap/valid/'
		self.test_folder    = self.input_folder + 'mocap/test/'
		self.filter_folder  = filter_folder
		self.filters_file   = filter_folder + 'motionDetector_step' + str(step) + '.npz'
		self.stats_file     = filter_folder + 'skeleton_stats'
		self.seq_per_class  = 500
		# network parameters
		self.input_size = [self.dlength * self.nframes, 0]
		self.batch_size     = self.seq_per_class * self.nclasses
		self.fc_layers = [400, self.nclasses]
		self.dropout_rates = [0.2, 0.2, 0.3]
		self.get_train_list()
		self.load_stats()





	def build_network(self, input_var=None):

		self.network= {}

		self.network['input'] = lasagne.layers.InputLayer(shape=(self.batch_size, self.input_size[0]),
										 input_var=self.x)


		# Add a fully-connected layer of 800 units, using the linear rectifier, and
		# initializing weights with Glorot's scheme (which is the default anyway):
		self.network['FC_1'] = batch_norm(lasagne.layers.DenseLayer(
				lasagne.layers.dropout(self.network['input'], p=self.dropout_rates[0]),  num_units=self.fc_layers[0],
				nonlinearity=lasagne.nonlinearities.tanh,
				W=lasagne.init.GlorotUniform()))


		# Finally, we'll add the fully-connected output layer, of 10 softmax units:
		self.network['prob'] = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(self.network['FC_1'], p=self.dropout_rates[1]), num_units=self.fc_layers[1],
				nonlinearity=lasagne.nonlinearities.softmax)


		# Each layer is linked to its incoming layer(s), so we only need to pass
		# the output layer to give access to a network in Lasagne:
		return self.network


	def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
		assert len(inputs) == len(targets)
		if shuffle:
			indices = numpy.arange(len(inputs))
			numpy.random.shuffle(indices)
		for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
			if shuffle:
				excerpt = indices[start_idx:start_idx + batchsize]
			else:
				excerpt = slice(start_idx, start_idx + batchsize)
			yield inputs[excerpt], targets[excerpt]



	def prepare(self, subset):

		if subset == 'train': folder = self.train_folder
		elif subset == 'valid': folder = self.valid_folder
		elif subset == 'test': folder = self.test_folder
		else: print "Unknown subset"

		mocap_dataset = numpy.zeros([self.seq_per_class * self.nclasses, self.nframes *self.dlength])

		labels = numpy.zeros((self.seq_per_class * self.nclasses,))

		sample = 0
		class_number = 0

		if subset == 'train':
			folder = self.train_folder
			file_list = self.train_list
		else:
			file_list = {}
			file_list[0] = glob.glob( folder + "*_g%02d*.pickle" %(0) )

			file_list[1] = []
			for cl in xrange(1,21):
				file_list[1] = file_list[1] + glob.glob( folder + "*_g%02d*.pickle" %(cl) )


		while sample < self.nclasses*self.seq_per_class:

			try:
				file_number = random.randint(0,len(file_list[class_number])-1)
				file_mocap = file_list[class_number][file_number]

				with open(file_mocap) as f:
					descr0,descr1 = cPickle.load(f)

				descr1 = numpy.subtract(descr1,self.mdt_mean2)
				descr1 = numpy.divide(descr1,numpy.sqrt(self.vdt_mean2))
				descr0 = numpy.subtract(descr0,self.mdt_mean1)
				descr0 = numpy.divide(descr0,numpy.sqrt(self.vdt_mean1))

				descr = numpy.concatenate((descr0,descr1),axis=1)

				if ((descr.shape[0]<60 and class_number==0 ) or class_number==1) and descr.shape[0]>=self.step*(self.nframes-1)+1:
					seq_number = random.randint(0, descr.shape[0]-self.step*(self.nframes-1)-1)
					mocap_dataset[sample] = descr[seq_number : seq_number+self.step*(self.nframes-1)+1 : self.step].flatten()

					labels[sample] = class_number
					class_number +=1
					if class_number == self.nclasses: class_number = 0

					sample += 1
			except:
				print 'Data not found (class ', class_number, ', folder ', folder, ')'
				class_number += 1
				if class_number == self.nclasses: class_number = 0


		if subset == 'train':
			self.labels_train = numpy.int8(labels)
			self.mocap_train = mocap_dataset.astype(theano.config.floatX)

		elif subset == 'valid':
			self.labels_valid = numpy.int8(labels)
			self.mocap_valid = mocap_dataset.astype(theano.config.floatX)

		elif subset == 'test':
			self.labels_test = numpy.int8(labels)
			self.mocap_test = mocap_dataset.astype(theano.config.floatX)




	def get_train_list(self):

		"""
		Function to retrieve list of training data filenames

		"""
		self.train_list = {}
		name_template = self.train_folder + "*_g%02d*.pickle"

		self.train_list[0] = glob.glob( name_template %(0) )

		self.train_list[1] = []
		for cl in xrange(1,21):
			self.train_list[1] = self.train_list[1] + glob.glob( name_template %(cl) )




	def load_stats(self):
		"""
		Function to load normalized data (zero-mean & variance)

		"""
		f = open(self.stats_file,'r')
		self.mdt_mean1 = cPickle.load(f)
		self.mdt_mean2 = cPickle.load(f)
		self.vdt_mean1 = cPickle.load(f)
		self.vdt_mean2 = cPickle.load(f)
		f.close()



	def train_lasagne(self, learning_rate_value=0.2, learning_rate_decay=0.9999, num_epochs=4000):
		# Load the dataset

		self.saved_params = []


		print "Loading data..."

		learning_rate = T.fscalar('learning_rate')
		epoch = T.fscalar('epoch')



		# Create neural network model (depending on first command line parameter)
		print "Building model and compiling functions..."

		self.network = self.build_network(self.x)

		# Create a loss expression for training, i.e., a scalar objective we want
		# to minimize (for our multi-class problem, it is the cross-entropy loss):
		prediction = lasagne.layers.get_output(self.network['prob'], deterministic=False)

		loss = lasagne.objectives.categorical_crossentropy(prediction, self.y)
		loss = loss.mean()
		# We could add some weight decay as well here, see lasagne.regularization.

		# Create update expressions for training, i.e., how to modify the
		# parameters at each training step. Here, we'll use Stochastic Gradient
		# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.

		params = lasagne.layers.get_all_params(self.network['prob'], trainable=True)

		updates = lasagne.updates.nesterov_momentum(
				loss, params,learning_rate, momentum=0.8)

		#updates = lasagne.updates.adam(loss,params,learning_rate=learning_rate_value, beta1=0.9, beta2=0.999, epsilon=1e-08)

		# Create a loss expression for validation/testing. The crucial difference
		# here is that we do a deterministic forward pass through the network,
		# disabling dropout layers.

		test_prediction = lasagne.layers.get_output(self.network['prob'], deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,self.y)
		test_loss = test_loss.mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.y),
						  dtype=theano.config.floatX)


		## Masking the gradients if masks are defined
		#if not self.mask_weights is None:
			#for param in self.params[-4:-2]:
				#if param.name == 'W':
					#updates[param] *= self.mask_weights
		#elif param.name == 'b':
			#updates[param] *= self.mask_biases


		# Compile a function performing a training ste on a mini-batch (by giving
		# the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([self.x, self.y, learning_rate], loss, updates=updates,on_unused_input='ignore')


		val_fn = theano.function([self.x , self.y], [test_loss, test_acc],on_unused_input='ignore')


		# Loading the training and validation set
		# Loading the validation set
		self.prepare('valid')


		n_train_batches = self.nclasses * self.seq_per_class / self.batch_size
		n_valid_batches = self.nclasses * self.seq_per_class / self.batch_size


		# Finally, launch the training loop.
		print "Starting training..."
		# We iterate over epochs:


		self.best_validation_acc = -numpy.inf
		done_looping = False



		for epoch in range(num_epochs):
			self.prepare('train')
		# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			for batch in self.iterate_minibatches(self.mocap_train, self.labels_train, self.batch_size, shuffle=True):
				inputs, targets = batch
				train_err += train_fn(inputs, targets,learning_rate = learning_rate_value)
				train_batches += 1

			# And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			for batch in self.iterate_minibatches(self.mocap_valid, self.labels_valid, self.batch_size, shuffle=False):
				inputs, targets = batch
				err, acc = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				val_batches += 1

			this_validation_acc = val_acc

			#self._print_results(this_validation_loss, ts, iter_,
			#learning_rate_value)

			if this_validation_acc > self.best_validation_acc:
				self.best_validation_acc = this_validation_acc

				f = open(self.filters_file, 'w')
				numpy.savez(f, *lasagne.layers.get_all_param_values(self.network['prob']))
				f.close()


			learning_rate_value = learning_rate_value * learning_rate_decay

			print "Epoch= %d Learning rate = %3.4f Validation Accuracy= %3.3f" %(epoch+1, learning_rate_value, val_acc/val_batches * 100)


		## After training, we compute and print the test error:
		#test_err = 0
		#test_acc = 0
		#test_batches = 0

		#self.prepare('test')
		#for batch in self.iterate_minibatches(self.mocap_test, self.labels_test, self.batch_size, shuffle=False):
			#inputs, targets = batch
			#err, acc = val_fn(inputs, targets)
			#test_err += err
			#test_acc += acc
			#test_batches += 1
		#print("Final results:")
		#print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
		#print("  test accuracy:\t\t{:.2f} %".format(
			#test_acc / test_batches * 100))

		# Optionally, you could now dump the network weights to a file like this:
		#numpy.savez('model.npz', *lasagne.layers.get_all_param_values(network))
		#
		# And load them again later on like this:
		# with numpy.load('model.npz') as f:
		#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		# lasagne.layers.set_all_param_values(network, param_values)





	#def train_lasagne(self, num_epochs=50):
		## Load the dataset



		#print("Loading data...")


		## Create neural network model (depending on first command line parameter)
		#print("Building model and compiling functions...")

		#network = self.build_network(self.x)


		## Create a loss expression for training, i.e., a scalar objective we want
		## to minimize (for our multi-class problem, it is the cross-entropy loss):
		#prediction = lasagne.layers.get_output(network, deterministic=False)

		#loss = lasagne.objectives.categorical_crossentropy(prediction, self.y)
		#loss = loss.mean()
		## We could add some weight decay as well here, see lasagne.regularization.

		## Create update expressions for training, i.e., how to modify the
		## parameters at each training step. Here, we'll use Stochastic Gradient
		## Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		#params = lasagne.layers.get_all_params(network, trainable=True)
		#updates = lasagne.updates.nesterov_momentum(
				#loss, params, learning_rate=0.2, momentum=0.9)

		## Create a loss expression for validation/testing. The crucial difference
		## here is that we do a deterministic forward pass through the network,
		## disabling dropout layers.

		#test_prediction = lasagne.layers.get_output(network, deterministic=True)
		#test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
																#self.y)
		#test_loss = test_loss.mean()
		## As a bonus, also create an expression for the classification accuracy:
		#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.y),
						  #dtype=theano.config.floatX)

		## Compile a function performing a training step on a mini-batch (by giving
		## the updates dictionary) and returning the corresponding training loss:
		#train_fn = theano.function([self.x, self.y], loss, updates=updates)

		## Compile a second function computing the validation loss and accuracy:
		#val_fn = theano.function([self.x, self.y], [test_loss, test_acc])



		## Loading the training and validation set


		#self.prepare('valid')

		## Finally, launch the training loop.
		#print("Starting training...")
		## We iterate over epochs:
		#for epoch in range(num_epochs):
			#self.prepare('train')
			## In each epoch, we do a full pass over the training data:
			#train_err = 0
			#train_batches = 0
			#start_time = time.time()
			#for batch in self.iterate_minibatches(self.mocap_train, self.labels_train, self.batch_size, shuffle=True):
				#inputs, targets = batch
				#train_err += train_fn(inputs, targets)
				#train_batches += 1

			## And a full pass over the validation data:
			#val_err = 0
			#val_acc = 0
			#val_batches = 0
			#for batch in self.iterate_minibatches(self.mocap_valid, self.labels_valid, self.batch_size, shuffle=False):
				#inputs, targets = batch
				#err, acc = val_fn(inputs, targets)
				#val_err += err
				#val_acc += acc
				#val_batches += 1

			 ##Then we print the results for this epoch:
			##print("Epoch {} of {} took {:.3f}s".format(
			    ##epoch + 1, num_epochs, time.time() - start_time))
			##print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			##print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			##print("  validation accuracy:\t\t{:.2f} %".format(
		        ##val_acc / val_batches * 100))

			#print(epoch+1, train_err/train_batches, val_err/val_batches * 100, val_acc/val_batches * 100)

		## After training, we compute and print the test error:
		#test_err = 0
		#test_acc = 0
		#test_batches = 0

		#self.prepare('test')
		#for batch in self.iterate_minibatches(self.mocap_test, self.labels_test, self.batch_size, shuffle=False):
			#inputs, targets = batch
			#err, acc = val_fn(inputs, targets)
			#test_err += err
			#test_acc += acc
			#test_batches += 1
		#print("Final results:")
		#print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
		#print("  test accuracy:\t\t{:.2f} %".format(
			#test_acc / test_batches * 100))

		## Optionally, you could now dump the network weights to a file like this:
		##numpy.savez('model.npz', *lasagne.layers.get_all_param_values(network))
		##
		## And load them again later on like this:
		## with numpy.load('model.npz') as f:
		##     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		## lasagne.layers.set_all_param_values(network, param_values)

	def set_filters(self, file_mocap = None):
		"""
		Function to load / set pretrained weights (if it exists) or 'filters'

		:type file_mocap: string
		:param file_mocap: file that stores weights(filters) of network trained using mocap data

		"""
		if file_mocap == None: file_mocap = self.filters_file


		with numpy.load(file_mocap) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.network['prob'], param_values)


	def load_sample(self, input):
		"""
		Function to load_sample from file and perform zero-mean and varaince normalization
		uses re.sub() - Find all substrings where the RE matches, and replace them with a different string

		"""
		if type(input) is str:

			file_name_mocap = re.sub('_r_', '_a_', input)
			file_name_mocap = re.sub('_l_', '_a_', file_name_mocap)
			file_name_mocap = re.sub('/color/', '/mocap/', file_name_mocap)
			file_name_mocap = re.sub('/depth/', '/mocap/', file_name_mocap)
			file_name_mocap = re.sub('/audio/', '/mocap/', file_name_mocap)
			file_name_mocap = re.sub('_color_', '_descr_', file_name_mocap)
			file_name_mocap = re.sub('_depth_', '_descr_', file_name_mocap)
			file_name_mocap = re.sub('_mocap_', '_descr_', file_name_mocap)
			file_name_mocap = re.sub('_audio_', '_descr_', file_name_mocap)

			with open(file_name_mocap, 'rb') as f: descr0, descr1  = cPickle.load(f)

		else:
			[descr0, descr1] = input



		# zero-mean, variance normalization...
		descr1 = numpy.subtract(descr1,self.mdt_mean2)
		descr1 = numpy.divide(descr1,numpy.sqrt(self.vdt_mean2))
		descr0 = numpy.subtract(descr0,self.mdt_mean1)
		descr0 = numpy.divide(descr0,numpy.sqrt(self.vdt_mean1))

		descr = numpy.concatenate((descr0,descr1),axis=1)

		min_length = len(descr)
		nseq = min_length - self.step * (self.nframes - 1)

		nseq_batch = int( numpy.ceil(float(nseq) / float(self.batch_size)) * self.batch_size )

		if nseq > 0:
			mocap_data = numpy.zeros([nseq_batch, self.nframes * self.dlength])

			for seq_number in xrange(nseq):
				mocap_data[seq_number] = descr[seq_number : seq_number + self.step * (self.nframes-1) + 1 : self.step].flatten()

		mocap_data = mocap_data.astype(theano.config.floatX)

		return mocap_data, nseq


	def test(self, test_sample):
		"""
		Function to test trained network on a sample. Returns probabilities of the sample belonging to a given class

		:type test_sample: string
		:param test_sample: filename of the test sample
		"""
		with numpy.load(self.filters_file) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			lasagne.layers.set_all_param_values(self.network['prob'], param_values)

		[mocap_data, nseq] = test_sample

		sample_length = len(mocap_data)
		n_batches = sample_length / self.batch_size

		get_probabilities = theano.function([self.x],  lasagne.layers.get_output(self.network['prob'], deterministic=True))

		probs = [get_probabilities(mocap_data[iii * self.batch_size: (iii + 1) * self.batch_size])
								  for iii in xrange(n_batches) ]

		return numpy.vstack(probs)[:nseq]


####################
##### TRAINING #####
####################

if __name__ == '__main__':


#########################################################################
	#print '3/9 Splitting into training and validation subsets...\n'
#########################################################################

	'''test_set = [42, 141, 240, 318, 399, 401, 419, 450, 691, 546, 592, 511, 672, 621, 630, 521, 569, 472, 477, 512, 544, 501, 612];
	name_template = folder_datasets + '*/train/' + '0%03d_*.pickle'
	for fn in test_set:
		for fl in glob.glob(name_template %fn):
			print fl
			dst_name = re.sub('train','valid',fl)
			shutil.move(fl,dst_name)

	test_set = [32, 102, 231, 259, 390, 412, 455];
	name_template = folder_datasets + '*/train/' + '0%03d_*.pickle'
	for fn in test_set:
		for fl in glob.glob(name_template %fn):
			print fl
			dst_name = re.sub('train','test',fl)
			shutil.move(fl,dst_name)'''


#########################################################################
	#print '6/9 Training an MLP on pose descriptors...'
#########################################################################
	#descriptor_size = 183

	folder_datasets = '/media/dhaneshr/Gipsy/Chalearn/data_preprocessedv2/realtest'
	filter_folder = './filters/'
	label_folder = './ground_truth/'
	prediction_folder = './predictions/'

	mc = motionDetector(number_of_classes = 2,step = 1,nframes = 5, input_folder = folder_datasets, filter_folder = folder_filters)

	mc.build_mlp()
	mc.train_mlp()


	#    train_skeleton_mlp(learning_rate=0.2, learning_rate_decay = 0.99999,#LR 0.4#    # for tanh learning_rate = 0.2, learning_rate_decay = 0.99997# for maxmargin 0.00001 and 0.9995
	#                           n_epochs=250000, batch_size=100*21, seq_per_class = 100,
	#                           descriptor_size = descriptor_size,
	#                           fileIDvalid = folder_datasets + 'mocap/valid/',#valid/',
	#                           fileIDtrain = folder_datasets + 'mocap/train/',#train/',
	#                           outputfileID = folder_filters + 'temp',
	#                           finetune = False, fileIDfilters = None,
	#                           stepp = 4)

	print ('Done.')

