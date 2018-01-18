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
theano.sandbox.cuda.use('gpu2') 


import lasagne

import cPickle as pickle
import gzip

from pylearn2.datasets.cifar10 import CIFAR10 
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
        discrete = True, H=1., N=1., **kwargs):
        
        self.discrete = discrete
        
        self.H = H 
        self.N = N 
        
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
		
        if self.discrete:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H*1.5,self.H*1.5)), **kwargs)
            # add the discrete tag to weights 			
            self.params[self.W]=set(['discrete'])
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
    

# This class extends the Lasagne Conv2DLayer to support Probabilistic Discretization of Weights
class Conv2DLayer(lasagne.layers.Conv2DLayer): # H determines the range of the weights [-H, H], and N determines the state number in discrete weight space of 2^N+1
    
    def __init__(self, incoming, num_filters, filter_size,
        discrete = True, H=1.,N=1., **kwargs):
        
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
        if a<0.85:
           c = 1
        else:
           c = 0
        
        b=random.random()
        state_rand = T.round(b*pow(2,N))*L-H #state_rand is a random state in the discrete weight space Z_N
        
        delta_W1 =c*(state_rand-parambest)#parambest would transfer to state_rand with probability of a, or keep unmoved with probability of 1-a
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
	n_samples = X.shape[0]
        indx = np.random.permutation(xrange(n_samples))
	    #this work
        for i in range(batches):
            sl = slice(i * batch_size, (i + 1) * batch_size)
            X_batch = X[indx[sl]]
            y_batch = y[indx[sl]]

            trans_1 = random.randint(0, (4*2))
            trans_2 = random.randint(0, (4*2))
            crop_x1 = trans_1
            crop_x2 = (32 + trans_1)
            crop_y1 = trans_2
            crop_y2 = (32 + trans_2)

            # flip left-right choice
            flip_lr = random.randint(0,1)

            # set empty copy to hold augmented images so that we don't overwrite
            X_batch_aug = np.copy(X_batch)
            # for each image in the batch do the augmentation
            for j in range(X_batch.shape[0]):
            # for each image channel
                for k in range(X_batch.shape[1]):
                # pad and crop images
                    img_pad = np.pad(X_batch_aug[j,k], pad_width=((4,4), (4,4)), mode='constant')
                    X_batch_aug[j,k] = img_pad[crop_x1:crop_x2, crop_y1:crop_y2]

                    # flip left-right if chosen   this problem should be paid more attention to this work
                    if flip_lr == 1:
                        X_batch_aug[j,k] = np.fliplr(X_batch_aug[j,k])

            # fit model on each batch
            #loss.append(train_fn(X_batch_aug, y_batch))
            new_loss = train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)
            loss += new_loss
            
        
        loss/=batches

        return loss
		
	# Test the network on the validation set
    def val_epoch(X,y):
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
	n_samples_valid = X.shape[0]
		
        for i in range(batches):
	    sl = slice(i * batch_size, (i + 1) * batch_size)
            X_batch_test = X[sl]
            y_batch_test = y[sl]
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100
        loss /= batches

        return err, loss      

    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
	# initialize the err to be 100%
    best_val_err = 100
    best_test_err = 100
    
    #initialize the best parameters
    best_epoch = 1
    best_params = lasagne.layers.get_all_params(network, discrete=True)
    best_update = 200 #intialize the update_type to be normal training
    
    verr = []
    tloss = []
	
    for epoch in range(num_epochs): 
        
		# if a new round of training did not search a better result for a long time, the network will transfer to a random state and continue to search
		# otherwise, the network will be normally trained
        if  epoch >= best_epoch + 10:
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
            best_epoch = epoch+1
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
    scio.savemat(path,{'valid_err':vr,'train_loss':tloss})
    
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
    alpha = .1 #0.1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
	
    batch_size = 1000 
    print("batch_size = "+str(batch_size))    
    
    # Training parameters
    num_epochs = 2000 
    print("num_epochs = "+str(num_epochs))
    

    activation = discrete_neuron_3states #activation discretization
    print("activation = discrete_neuron")
	
    discrete = True 
    print("discrete = "+str(discrete))
    
    global update_type,best_params,H,N,th

    H = 1. # the weight is in [-H, H]
    print("H = "+str(H))
    N = 1. # the state number of the discrete weight space is 2^N+1
    print("N = "+str(N)+" Num_States = "+str(pow(2,N)+1))
    th = 3. #the nonlinearity parameter of state transfer probability
    print("tanh = "+str(th))

    # Decaying LR 
    LR_start = 0.01         #0.01
    print("LR_start = "+str(LR_start))
    LR_fin = 0.00003      # 0.0000003 
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))

    print('Loading CIFAR10 dataset...')
	
    train_set_size = 45000 
    train_set = CIFAR10(which_set="train",start=0,stop = train_set_size)
    valid_set = CIFAR10(which_set="train",start=train_set_size,stop = 50000)
    test_set = CIFAR10(which_set="test")
        
        
    # bc01 format
    # Inputs in the range [-1,+1]
    # print("Inputs in the range [-1,+1]")
    '''
    train_set.X = train_set.X.reshape(-1,3,32,32)
    valid_set.X = valid_set.X.reshape(-1,3,32,32)
    test_set.X = test_set.X.reshape(-1,3,32,32)
    '''
    train_set.X = np.reshape(np.subtract(np.multiply(2./255,train_set.X),1.),(-1,3,32,32))
    valid_set.X = np.reshape(np.subtract(np.multiply(2./255,valid_set.X),1.),(-1,3,32,32))
    test_set.X = np.reshape(np.subtract(np.multiply(2./255,test_set.X),1.),(-1,3,32,32))
     
    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)
    
    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])
    
    # for hinge loss
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
            shape=(None, 3, 32, 32),
            input_var=input)

	# 128C3-128C3-P2
    cnn = Conv2DLayer(
            cnn, 
            discrete=discrete,
            H=H,
	    N=N,
            num_filters=128, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
   
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
            num_filters=128, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2)) 
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
              
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
			
    
	
    # 256C3-256C3-P2             
    cnn = Conv2DLayer(
            cnn, 
            discrete=discrete,
            H=H,
	    N=N,
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)

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
            num_filters=256, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
              
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
      
	
    # 512C3-512C3-P2              
    cnn = Conv2DLayer(
            cnn, 
            discrete=discrete,
            H=H,
	    N=N,
            num_filters=512, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    
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
            num_filters=512, 
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
               
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
   
    
    # 1024FC-10FC            
    cnn = DenseLayer(
                cnn, 
                discrete=discrete,
                H=H,
		N=N,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=1024)    
                  
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
                nonlinearity=lasagne.nonlinearities.identity,  #identity
                num_units=10)   
                      
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

