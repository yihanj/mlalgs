__author__ = 'yihanjiang'
import numpy
import scipy.optimize as opt

def sigmoid(X):
    return 1 / (1 + numpy.exp(- X))

def cost(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta)) # predicted probability of label 1
    log_l = (-y)*numpy.log(p_1) - (1-y)*numpy.log(1-p_1) # log-likelihood vector

    return log_l.mean()

def grad(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta))
    error = p_1 - y # difference between label and prediction
    grad = numpy.dot(error, X_1) / y.size # gradient vector

    return grad

# prefix an extra column of ones to the feature matrix (for intercept term)
y = [1 for _ in xrange(100)]
X = y

theta = 0.1* numpy.random.randn(3)
X_1 = numpy.append( numpy.ones((X.shape[0], 1)), X, axis=1)

theta_1 = opt.fmin_bfgs(cost, theta, fprime=grad, args=(X_1, y))