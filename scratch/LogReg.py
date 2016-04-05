'''
This code is for MNIST handwritten digits classification.
Uses a stochastic gradient descent optimization method suitable for large datasets.
Adpated from www.deeplearning.net/tutorial
It is parametrized by a weight matrix W: and bias vector b: Classification is done by projecting the datapoints 
onto a set of hyperplanes, the distance to which is used to determine a class memebership probability.

'''
import theano
import numpy as np
from theano import function
import theano.tensor as T
import os
import cPickle
import gzip
import sys
import time

class LogisticRegression(object):
    ''' The main class that deals wit regression problem'''

    def __int__(self,input,n_in,n_out):
        '''
            intitializing the parameter for logistic regression
            initializing the Weight matrix with zero of shape (n_in,n_out)
            n_in = number of input units -> datapoints
            n_out= number of output units -> labels
            '''
        self.W = theano.shared(value = np.zeros((n_in,n_out), dtype=theano.config.floatX), name = 'W', borrow = True)
        '''
            initializing the bias balues b
        '''
        self.b = theano.shared(value = np.zeros((n_out), dtype=theano.config.floatX), name = 'b', borrow = True)
# computing the matrix of class memebership for a particular sample
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
# computing the maximum probability to a particular class
        self.y_predict =  T.argmax(self.p_y_given_x, axis = 1)
# Model parameters
        self.params = [self.w, self.b]
# Returns the mean of negative log-likelihood of prediction of this model under a given target distribution
        def negative_log_likelihood(self,y):
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
# Defining the error in prediction
        def errors(self,y):
# will return a float representing the number of errors over the total number of examples of minibatches
            if y.ndim != self.y_pred.ndim:
                raise TypeError('y should have the same shape as self.y_pred', ('y',y.type, 'y_pred', self.y_pred.type))
# Checking ify is of correct datatype
            if y.dtype.startwith('int'):
                return T.mean(T.neq(self.y_pred,y))
            else:
                raise NotImplementedError()
# Loading the dataset here
def load_dataset(dataset):
    '''Loading the dataset '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join( os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print 'Downloading the mnist dataset from %s' % origin
        urllib.urlretrieve(origin, dataset)
    print '...loading data'
# Load the dataset now
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_test = cPickle.load(f)
    f.close()
# Creating shared variables
    def shared_dataset(data_xy, borrow = True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype = theano.config.floatX), borrow=borrow)
        '''returning the splitted shared variable'''
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    '''returning the final rvalue in single variable'''
    rval = [(train_set_x,train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

# Stochastic Gradient descent optimization of a log-linear model
def sgd_optimization_mnist(learning_rate = 0.13, n_epochs = 1000, dataset = 'mnist.pkl.gz', batch_size = 600):
    datasets = load_dataset(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
#computing the number of minibatches of each type of set
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    '''Building the model Here'''
    print '....Building Model'
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    '''Creating Classifier object'''
    Classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    '''Cost to minimize -> negative log likelihood function'''
    cost = classifier.negative_log_likelihood(y)
    ''' compliling the theano functions'''
    test_model = theano.function(inputs = [index], outputs = classifier.errors(y), givens = { x: test_set_x[index *batch_size: (index+1)*batch_size], y: test_set_y[index*batch_size: (index+1)*batch_size]})
    validate_model = theano.function (inputs = [index], outputs = classifier.errors(y), givens = {x: valid_set_x[index*batch_size: (index+1)*batch_size], y: valid_set_y[index*batch_size: (index+1)* batch_size]})
    '''computing the gradients of the cost with respect to theta'''
    g_w = T.grad(cost=cost, wrt= classifier.W)
    g_b = T.grad(cost=cost, wrt= classifier.b)
    ''' Defining the updates of Weights -> W and bias values -> b'''
    updates = [(classifier.W, calssifier.W - learning_rate * g_W), (classifier.b, classifier.b - learning_rate * g_b)]

    '''Compiling the train-model now'''
    train_model = theano.function( inputs=[index], outpust=cost, updates= updates, givens= { x: train_set_x[index*batch_size: (index+1)*batch_size], y: train_set_y[index*batch_size:(index+1)*batch_size]})

#Training the Model 
    print '... Training the Model Now'
    patience=5000
    patience_increase =2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience /2)
    best_validation_loss= numpy.inf
    test_score=0.
    start_time = time.clock()
    done_looping = False
    epoch =0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch +1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch-1) * n_train_batches + minibatch_index
            if (iter+1) % validation_frequency == 0 :
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index+1, n_train_batches, this_validation_loss*100.))
                if (this_validation_loss < best_validation_loss):
                    if this_validation_loss < best_validation_loss * \
                        improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_lss = this_validation_loss
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('epoch %i, minibatch %i/%i, test error of' ' best model %f %%' ) % (epoch, minibatch_index+1, n_train_batches,test_score*100.))
                if patience <= iter:
                    done_looping = True
                    break
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,' 'with test performance %f %%' ) % (best_validation_loss* 100., test_score* 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1.*epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file' + os.path.split(__file__)[1]+ 'ran for %.1fs' % ((end_time - start_time)))
if __name__ == '__main__':
    sgd_optimization_mnist()
