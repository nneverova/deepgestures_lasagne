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
import random
import numpy.ma
import math
import re
import cPickle
import glob

import theano
import theano.tensor as T


import lasagne
from lasagne.layers import batch_norm

from basicClassifier_lasagne import basicClassifier

class audioClassifier(basicClassifier):

    """ Gesture recognition based on audio samples"""

    def __init__(self, input_folder, filter_folder, number_of_classes=21, step=4, nframes=5, batch_size=42, pretrained=False, do_lmdb = False):
        basicClassifier.__init__(self, input_folder, filter_folder, number_of_classes, step, nframes, batch_size, 'audio', pretrained,)

                        # input parameters
        self.input_size['audio'] = [9, 40]
        self.modality_list = ['audio']
        self.hand_list['both'] = self.modality_list
        self.network= {}
        # network parameters
        self.batch_size = 42
        self.conv_layers = [(25,1,5,5)]
        self.pooling = [(1,1)]
        self.fc_layers = [700, 350, self.nclasses]
        self.dropout_rates = [0.0, 0.0, 0.0, 0.0] # dropout rates for fully connected layers
        self.activations = [self.activation] * (len(self.conv_layers) + len(self.fc_layers) - 1)

        # training parameters
        self.learning_rate_value = 0.05
        self.learning_rate_decay = 0.9999
        self.n_epochs = 1500

        lasagne.random.set_rng(numpy.random.RandomState(1234))  # a fixed seed to reproduce results

        # Theano variables
        tensor4 = T.TensorType(theano.config.floatX, (False,) * 4)
        self.sinputs = [T.tensor4('x')]



    def build_network(self, input_var=None):
        if not input_var is None: self.sinputs = input_var

        self.network['input'] = lasagne.layers.InputLayer(shape=(self.batch_size, 1, self.input_size['audio'][0],self.input_size['audio'][1]),
                                                          input_var=self.sinputs[0])

        self.network['Conv2D_1'] = batch_norm(lasagne.layers.Conv2DLayer(
            lasagne.layers.dropout(self.network['input'], p=self.dropout_rates[0]) , num_filters=25, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform()))

        self.network['MaxPool2D_1'] = lasagne.layers.MaxPool2DLayer(self.network['Conv2D_1'], pool_size=(1, 1))

        self.network['FC_1'] = batch_norm(lasagne.layers.DenseLayer(
            lasagne.layers.dropout(self.network['MaxPool2D_1'], p=self.dropout_rates[1]),
            num_units=self.fc_layers[0],
            nonlinearity=lasagne.nonlinearities.tanh))

        self.network['FC_N'] = batch_norm(lasagne.layers.DenseLayer(lasagne.layers.dropout(self.network['FC_1'], p=self.dropout_rates[2]),
            num_units=self.fc_layers[1],
            nonlinearity=lasagne.nonlinearities.tanh))


        self.network['prob'] =  batch_norm(lasagne.layers.DenseLayer(
            lasagne.layers.dropout(self.network['FC_N'], p=self.dropout_rates[3]),
            num_units=self.nclasses,
            nonlinearity=lasagne.nonlinearities.softmax))

        return self.network




    def train_network(self, num_epochs=1500):
        # Load the dataset

        print("Loading data...")

        y = T.ivector('y')
        learning_rate = T.fscalar('learning_rate')
        epoch = T.fscalar('epoch')

        # Create neural network model (depending on first command line parameter)
        print("Building model and compiling functions...")


        self.network = self.build_network(self.sinputs)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(self.network, deterministic=False)

        loss = lasagne.objectives.categorical_crossentropy(prediction, y)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, self.learning_rate_value, momentum=0.8)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.

        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,y)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
                          dtype=theano.config.floatX)


        ## Masking the gradients if masks are defined
        #if not self.mask_weights is None:
                        #for param in self.params[-4:-2]:
                                        #if param.name == 'W':
                                                        #updates[param] *= self.mask_weights
        #elif param.name == 'b':
                        #updates[param] *= self.mask_biases


        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([self.sinputs, y], loss, updates=updates)

        #train_fn = theano.function(self.sinputs + [y, learning_rate, epoch],
                                                                                                                            # loss, updates=updates, on_unused_input='ignore')

        # Compile a second function computing the validation loss and accuracy:
        #val_fn = theano.function(self.sinputs + [y, learning_rate, epoch], [test_loss, test_acc])
        val_fn = theano.function([self.sinputs,y], [test_loss, test_acc])


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
        for epoch in range(num_epochs):
            self._load_dataset('train')

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for minibatch_index in xrange(n_train_batches):
                train_inputs = self._prepare_inputs('train',minibatch_index)
                #print train_inputs[0].shape
                #print train_inputs[1].shape
                train_err += train_fn(*train_inputs)
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

                #Then we print the results for this epoch:
            #print("Epoch {} of {} took {:.3f}s".format(
                #epoch + 1, num_epochs, time.time() - start_time))
            #print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            #print("  validation accuracy:\t\t{:.2f} %".format(
                #val_acc / val_batches * 100))

            print(epoch+1, train_err/train_batches, val_err/val_batches, val_acc/val_batches * 100)

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
        numpy.savez('model.npz', *lasagne.layers.get_all_param_values(self.network))
        #
        # And load them again later on like this:
        # with numpy.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)


    def _load_file(self, file_name, data_sample=None):
        #file_name = re.sub('/mnt/data/dramacha/data_preprocessedv2/','/mnt/data/datasets/montalbano/data_preprocessed/',file_name)
        for mdlt in ['/depth/','/color/','/mocap/']:
	    if("realtest" in file_name):
		audio_dir = '/audio/'
	    else:
		audio_dir = '/audio_cleaned/'
	    file_name = re.sub(mdlt,audio_dir,file_name) #edit to rename 'audio_cleaned' when training
        for hnd in ['_r_','_l_']:
            file_name = re.sub(hnd,'_a_',file_name)
        for suff in ['depth','color','descr']:
            file_name = re.sub(suff,'audio',file_name)
        if not os.path.isfile(file_name):
            #print file_name
            file_name = glob.glob(file_name[:-14]+'*')[0]
        if data_sample is None:
            data_sample = {}
        for hnd in self.hand_list:
            if not hnd in data_sample:
                data_sample[hnd] = {}
            for mdlt in self.modality_list:
                with open(file_name,'rb') as f:
                    [data_sample[hnd][mdlt]] = cPickle.load(f)
        if not 'min_length' in data_sample:
            data_sample['min_length'] = len(data_sample[hnd][mdlt])/2
        else:
            data_sample['min_length'] = min(data_sample['min_length'],len(data_sample[hnd][mdlt])/2)
        return data_sample

    def _get_stblock(self, data_input, hnd, mdlt, start_frame=None):
        if start_frame is None:
            start_frame = random.randint(0, data_input['min_length']-self.step*(self.nframes-1)-1)
        end_ind = (start_frame+self.step*(self.nframes-1)+1)*2
        stblock = data_input[hnd][mdlt][start_frame*2 : end_ind : self.step] * 20.
        return stblock, True
