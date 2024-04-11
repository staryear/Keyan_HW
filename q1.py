# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import csv

def gass_hermite_quad( f, degree):
    '''
    Calculate the integral (1) numerically.
    :param f: target function, takes a array as input x = [x0, x1,...,xn], and return a array of function values f(x) = [f(x0),f(x1), ..., f(xn)]
    :param degree: integer, >=1, number of points
    :return:
    '''
    points, weights = np.polynomial.hermite.hermgauss( degree)	# points.shape = weights.shape = (degree,)
    #function values at given points
    f_x = f( points)	# f_x.shape = (degree,)
    #weighted sum of function values
    F = np.sum( f_x  * weights)
    return F# Press Ctrl+F8 to toggle the breakpoint.

def ques1_sigmoid(x):
    # returns sigmoid(10x + 3) for function exp(-x^2)*sigmoid(10x+3)
    return(expit(10*x + 3))

def gaussian(mean, std, x):
    return(norm.pdf(x, loc = mean, scale=std))


def compute_gauss_hermite_approx(x):

    F = gass_hermite_quad(ques1_sigmoid, degree=100);

    print("when degree is 100, the normalization constant value is", F)

    return (np.exp(-x*x)*expit(10*x + 3)/F)


def neg_log_ques1(x):
    y = -x * x + np.log(expit(10 * x + 3))
    return (-y)

def get_MAP():

    res = minimize(neg_log_ques1, np.array(0))
    map = res.x
    return(map)

def compute_laplace_approx(x):

    mean = get_MAP()
    sigmoid = expit(10*mean+3)
    var =  1/(2 + 100 * sigmoid*(1-sigmoid))
    y = gaussian(mean, math.sqrt(var), x)
    print("mean is", mean)
    print("variance is", var)
    return(y)

def get_lambda(xi):
    lamb = -(1/(2*(xi*10+3)))*(expit(10*xi+3) - 0.5)
    return lamb

def compute_var_local_inference(x):

    xi = 0
    degree = 100
    Z1 = gass_hermite_quad(ques1_sigmoid, degree)
    def get_sigmod_y(x):

        lamb = get_lambda(xi)
        sigmoid_y = expit(10*xi+3) * np.exp(5 * (x - xi) + lamb * np.multiply(10*(x-xi), 10*(x+xi)+6))
        return sigmoid_y

    for i in range(100):
        qx = get_sigmod_y(x)*np.exp(-x*x)/Z1
        xi = x[np.argmax(qx)]
    print("the final xi is ", xi)
    return qx

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("question1(a)")
    x = np.linspace(-5, 5, 500)
    y = compute_gauss_hermite_approx(x)
    #print(x,y)
    plt.plot(x, y)
    plt.show()

    y_2 = compute_laplace_approx(x)
    plt.plot(x, y_2)
    plt.show()

    y_3 = compute_var_local_inference(x)
    print("mean is", y_3.mean())
    print("variance is", y_3.var())
    plt.plot(x, y_3)
    plt.show()
    plt.plot(x, y)
    plt.plot(x, y_2)
    plt.plot(x, y_3)
    plt.legend(["Gauss Hermite", "Laplace Approx", "Local Variational"])
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
