# The basic code framework is based on the BinaryNet (https://github.com/MatthieuCourbariaux/BinaryNet/blob/master/Train-time/binary_net.py)
# We mainly modify the gradient calculation (e.g. discrete_grads function) and neuronal activition (e.g. discrete_neuron_3states) for network training. 
# And we save the best parameters for searching a better result.
# For multilevel extension, you can simply modify the activation function and the N parameter for weight.
# Please cite our paper if you use this code: https://arxiv.org/pdf/1705.09283.pdf

from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility


import theano
import theano.tensor as T
# specifying the gpu to use
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1') 

import lasagne

import cPickle as pickle
import gzip

from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict

import time

import numpy as np

from theano.ifelse import ifelse

import matplotlib.pyplot as plt #for drawing
import scipy.io as scio
from numpy import random
from numpy import multiply

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
from itertools import izip
class round_custom(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round_scalar = round_custom(same_out_nocomplex, name='round_var')
round_var = Elemwise(round_scalar)


def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)	

def discrete_neuron_3states(x): #discrete activation with three states
     return T.cast(round_var(hard_sigmoid(2*(x-1))+hard_sigmoid(2*(x+1))-1 ),theano.config.floatX)

# This class extends the Lasagne DenseLayer to support Probabilistic Discretization of Weights
class DenseLayer(lasagne.layers.DenseLayer): # H determines the range of the weights [-H, H], and N determines the state number in discrete weight space of 2^N+1
    
    def __init__(self, incoming, num_units, 
        discrete = True, H=1.,N=1., **kwargs): 
        
        self.discrete = discrete
        
        self.H = H
        self.N = N
        
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        if self.discrete:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the discrete tag to weights            
            self.params[self.W]=set(['discrete'])
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
    

# This class extends the Lasagne Conv2DLayer to support Probabilistic Discretization of Weights
class Conv2DLayer(lasagne.layers.Conv2DLayer): # H determines the range of the weights [-H, H], and N determines the state number in discrete weight space of 2^N+1
    
    def __init__(self, incoming, num_filters, filter_size,
        discrete = True, H=1.,N=1.,**kwargs):
        
        self.discrete = discrete
        
        self.H = H 
        self.N = N 
                  
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.discrete:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the discrete tag to weights            
            self.params[self.W]=set(['discrete'])
            
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)   

#fine tuning of weight element to locate at the neareast 2^N+1 descrete states in [-H, H] 
def weight_tune(W,l_limit,r_limit):
    global N
    state_index = T.cast(T.round((W-l_limit)/(r_limit-l_limit)*pow(2,N)),theano.config.floatX)
    W = state_index/pow(2,N)*(r_limit-l_limit) + l_limit

    return W
	
#discrete the delta_W from real value to be k*L, where k is an integer and L is the length of state step, i.e. 2H/(2^N)
def discrete_grads(loss,network,LR):
    global update_type,best_params,H,N,th # th is a parameter that controls the nonlinearity of state transfer probability

    W_params = lasagne.layers.get_all_params(network, discrete=True) #Get all the weight parameters
    layers = lasagne.layers.get_all_layers(network)
	
    W_grads = []
    for layer in layers:
        params = layer.get_params(discrete=True)
        if params:
            W_grads.append(theano.grad(loss, wrt=layer.W)) #Here layer.W = weight_tune(param) 
    updates = lasagne.updates.adam(loss_or_grads=W_grads,params=W_params,learning_rate=LR)  
	
    for param, parambest in izip(W_params, best_params) :

        L = 2*H/pow(2,N) #state step length in Z_N 
		
        a=random.random() #c is a random variable with binary value
        if a<0.8:
           c = 1
        else:
           c = 0
        
        b=random.random()
        state_rand = T.round(b*pow(2,N))*L - H #state_rand is a random state in the discrete weight space Z_N
        
        delta_W1 =c*(state_rand-parambest) #parambest would transfer to state_rand with probability of a, or keep unmoved with probability of 1-a
        delta_W1_direction = T.cast(T.sgn(delta_W1),theano.config.floatX)
	dis1=T.abs_(delta_W1) #the absolute distance
        k1=delta_W1_direction*T.floor(dis1/L) #the integer part
        v1=delta_W1-k1*L #the decimal part
        Prob1= T.abs_(v1/L) #the transfer probability
	Prob1 = T.tanh(th*Prob1) #the nonlinear tanh() function accelerates the state transfer
		   
        delta_W2 = updates[param] - param 
        delta_W2_direction = T.cast(T.sgn(delta_W2),theano.config.floatX)	   
        dis2=T.abs_(delta_W2) #the absolute distance
        k2=delta_W2_direction*T.floor(dis2/L) #the integer part
        v2=delta_W2-k2*L #the decimal part
        Prob2= T.abs_(v2/L) #the transfer probability
        Prob2 = T.tanh(th*Prob2) #the nonlinear tanh() function accelerates the state transfer
        
        srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        Gate1 = T.cast(srng.binomial(n=1, p=Prob1, size=T.shape(Prob1)), theano.config.floatX) # Gate1 is a binary variable with probability of Prob1 to be 1
        Gate2 = T.cast(srng.binomial(n=1, p=Prob2, size=T.shape(Prob2)), theano.config.floatX) # Gate2 is a binary variable with probability of Prob2 to be 1

        delta_W1_new=(k1+delta_W1_direction*Gate1)*L #delta_W1_new = k*L where k is an integer   
        updates_param1 = T.clip(parambest + delta_W1_new,-H,H)
        updates_param1 = weight_tune(updates_param1,-H,H) #fine tuning for guaranteeing each element strictly constrained in the discrete space

        delta_W2_new=(k2+delta_W2_direction*Gate2)*L #delta_W2_new = k*L where k is an integer  
        updates_param2 = T.clip(param + delta_W2_new,-H,H)
        updates_param2 = weight_tune(updates_param2,-H,H) #fine tuning for guaranteeing each element strictly constrained in the discrete space

	# if update_type<100, the weight probabilistically tranfers from parambest to state_rand, which helps to search the global minimum
        # elst it would probabilistically transfer from param to a state nearest to updates[param]		
        updates[param]= T.switch(T.lt(update_type,100), updates_param1, updates_param2)    
    
    return updates


def train(  network,
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test):    
    
    global update_type,best_params,H,N,th
    # A function which shuffles a dataset
    def shuffle(X,y):
    
        shuffled_range = range(len(X))
        np.random.shuffle(shuffled_range)
        
        new_X = np.copy(X)
        new_y = np.copy(y)
        
        for i in range(len(X)):           
            new_X[i] = X[shuffled_range[i]]
            new_y[i] = y[shuffled_range[i]]
            
        return new_X,new_y
    
    #train the network for one epoch on the training set 
    def train_epoch(X,y,LR):    
        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            new_loss = train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
            loss += new_loss
            
        
        loss/=batches

        return loss
    
    # Test the network on the validation set
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100 
        loss /= batches

        return err, loss               
    
    # shuffle the training set
    X_train,y_train = shuffle(X_train,y_train)
	# initialize the err to be 100%
    best_val_err = 100
    best_test_err = 100
	
	#initialize the best parameters
    best_epoch = 1
    best_params = lasagne.layers.get_all_params(network, discrete=True)
    update_type = 200 #intialize the update_type to be normal training
	
    verr = []
    tloss = []
    
    for epoch in range(num_epochs): 
        
		# if a new round of training did not search a better result for a long time, the network will transfer to a random state and continue to search
		# otherwise, the network will be normally trained
        if  epoch >= best_epoch + 15:
	        update_type = 10       
        else:
            update_type = 200 
        
        if epoch==0: # epoch 0 is for weight initialization to a discrete space Z_N without update
            LR = 0
        elif epoch<=1:
            LR = LR_start

        else:
            LR = LR*LR_decay #decay the LR  

        start_time = time.time()


        train_loss = train_epoch(X_train,y_train,LR)
        
        X_train,y_train = shuffle(X_train,y_train)
        
        val_err, val_loss = val_epoch(X_val,y_val)
        test_err, test_loss = val_epoch(X_test,y_test)
		
        if epoch>=1: #collect data for plot
            tloss.append(train_loss)
            verr.append(val_err)
	       
        if test_err <= best_test_err:            
            best_test_err = test_err
            best_epoch = epoch + 1
            best_params = lasagne.layers.get_all_params(network, discrete=True)
	
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  update_type:                   "+str(update_type)) 
        print("  LR:                            "+str(LR))
        print("  th:                            "+str(th))
        print("  LR_decay:                      "+str(LR_decay))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best test error rate:          "+str(best_test_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        
        
     
    path = 'H'+str(H)+'N'+str(N)+'LR'+str(LR_start)+'D'+str(LR_decay)+'B'+str(batch_size)+'E'+str(num_epochs)+'tanh'+str(th)+'.mat'
    scio.savemat(path,{'valid_err':verr,'train_loss':tloss})
    
    fig = plt.figure(1) 
    x = np.arange(num_epochs-1) + 1
    sub1 = fig.add_subplot(211) 
    line1 = sub1.plot(x,verr,'r-',linewidth=2) 
    plt.xlabel('training epoch')
    plt.ylabel('validation error rate')
    sub2 = fig.add_subplot(212)
    line2 = sub2.plot(x,tloss,'b-',linewidth=2) 
    plt.xlabel('training epoch')
    plt.ylabel('training_loss')
    
    plt.show()

    

if __name__ == "__main__":
    
	# BN parameters
    alpha = 0.1 
    print("alpha = "+str(alpha))
    epsilon = 1e-4 
    print("epsilon = "+str(epsilon))
	
    batch_size = 10000 
    print("batch_size = "+str(batch_size))
       
    # Training parameters
    num_epochs = 4000 
    print("num_epochs = "+str(num_epochs))
    

    activation = discrete_neuron_3states #activation discretization
    print("activation = discrete_neuron_3states")
	
    discrete = True
    print("discrete = "+str(discrete))
    
    global update_type,best_params,H,N,th

    H = 1. # the weight is in [-H, H]
    print("H = "+str(H))
    N = 1. # the state number of the discrete weight space is 2^N+1
    print("N = "+str(N)+" Num_States = "+str(pow(2,N)+1))
    th = 3.   #the nonlinearity parameter of state transfer probability
    print("tanh = "+str(th))

    
    # Decaying LR 
    LR_start = 0.1 
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000001 
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./(num_epochs)) 
    print("LR_decay = "+str(LR_decay))

    print('Loading MNIST dataset...')
    
    train_set = MNIST(which_set= 'train', start=0, stop = 50000, center = False)
    valid_set = MNIST(which_set= 'train', start=50000, stop = 60000, center = False)
    test_set = MNIST(which_set= 'test', center = False)
    
    train_set.X = 2*train_set.X.reshape(-1, 1, 28, 28)-1.
    valid_set.X = 2*valid_set.X.reshape(-1, 1, 28, 28)-1.
    test_set.X = 2*test_set.X.reshape(-1, 1, 28, 28)-1.

    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')

    LR = T.scalar('LR', dtype=theano.config.floatX)
    
    update_type = 200 #intialize the update_type to be normal training

    cnn = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input) 
    
    cnn = Conv2DLayer(
            cnn, 
            discrete=discrete,
            H=H,
            N=N,
            num_filters=32, 
            filter_size=(5, 5),
            pad = 'valid',
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
	
    cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
				
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
			
    cnn = Conv2DLayer(
            cnn, 
            discrete=discrete,
            H=H,
            N=N,
            num_filters=64, 
            filter_size=(5, 5),
            pad = 'valid',
            nonlinearity=lasagne.nonlinearities.identity)

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
	
    cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
				
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
    
    cnn = DenseLayer(
                cnn, 
                discrete=discrete,
                H=H,
                N=N,
                num_units=512,
                nonlinearity=lasagne.nonlinearities.identity) 
    
    cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
 
    cnn = DenseLayer(
                cnn, 
                discrete=discrete,
                H=H,
                N=N,
                num_units=10,
                nonlinearity=lasagne.nonlinearities.identity) 
    cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    best_params = lasagne.layers.get_all_params(cnn, discrete=True)
	
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))

    if discrete:  
        updates = discrete_grads(loss,cnn,LR)
        params = lasagne.layers.get_all_params(cnn, trainable=True, discrete=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)


    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    train_fn = theano.function([input, target, LR], loss, updates=updates)
    val_fn = theano.function([input, target], [test_loss, test_err])

    print('Training...')
    
    train(  cnn,
            train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y)

