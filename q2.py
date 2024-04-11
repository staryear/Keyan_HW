import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import csv

def norm_data(x):
	#To normalise the data within [-1, 1]
	x =  (x - np.mean(x, axis=0))*(1/(np.max(x,axis=0) - np.min(x, axis=0)))    #max(x) = 0.47, min(x) = -0.53
	return(x)

def get_data(filename):
    x = []
    y = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i in csv_reader:
            x.append(i[0:-1])
            y.append(i[-1])
    x = np.array(x).astype(np.float32)
    x = np.hstack((x, np.ones((x.shape[0], 1))))  # for the bias term
    y = np.array(y).astype(np.float32)
    x[:, :-1] = norm_data(x[:, :-1])
    return(x,y)

def compute_map(x, y, w0):
    def log_posterior(w):
        global x, y
        w = w.reshape((x.shape[1],1))
        y = y.reshape(y.shape[0],1)
        temp = expit(np.matmul(x, w))
        pos = np.sum(y*np.log(temp) + (1-y)*np.log(1-temp)) - np.matmul(w.transpose(), w)/2
        return(-pos)
    print(w0.shape)
    return(minimize(log_posterior, w0))

def get_precision(x, w_map):
    sigmoid = expit(np.matmul(x, w_map))
    prec = np.eye(x.shape[1])
    for i in range(x.shape[0]):
        temp1 = x[i].reshape(x[i].shape[0], 1)
        prec = prec + np.matmul(temp1, temp1.transpose()) * sigmoid[i] * (1 - sigmoid[i])
    return prec

def laplace_approx(x,y):

    res = compute_map(x, y.reshape(y.shape[0], 1), np.zeros((x.shape[1])))
    w_map = res.x
    prec = get_precision(x, w_map)
    cov = np.linalg.inv(prec)
    return w_map, cov

def test(w, X, y):

    n,d = X.shape
    mu = expit(np.matmul(X,w))
    yhat = np.zeros((n,1)).astype(np.float32)
    yhat[mu>0.5]=1
    yhat = yhat.reshape(y.shape)
    correct = np.sum(yhat==y)
    return(correct,n)

def gauss_hermite(mean, std):
    """Compute the Gaussian-Hermite integral of the sigmoid function."""
    degree = 100
    points, weights = np.polynomial.hermite.hermgauss(degree)
    adjusted_points = points * np.sqrt(2) * std + mean
    val = expit(adjusted_points)
    F = np.dot(val, weights)  # Use dot product for efficiency
    return F

def predictive_likelihood(x_test, y_test, m, s):
    """Calculate the predictive likelihood of test data given model parameters."""
    likelihood = 0.0
    m = m.reshape(-1, 1)  # Ensure m is a column vector

    for i, x_i in enumerate(x_test):
        x_i = x_i.reshape(-1, 1)  # Reshape x_i as a column vector
        mu = x_i.T @ m
        std = np.sqrt(x_i.T @ s @ x_i)
        prob_1 = gauss_hermite(mu, std) * (1 / np.sqrt(np.pi))
        likelihood += prob_1 * y_test[i] + (1 - prob_1) * (1 - y_test[i])

    return likelihood / len(x_test)


def variational_logistic_meanfield(x, y):
    """Perform variational logistic regression using mean-field approximation."""
    max_iter = 100
    xi = -np.ones(x.shape[0])
    m = np.ones(x.shape[1])
    s = np.zeros((x.shape[1], x.shape[1]))

    for _ in range(max_iter):
        s = compute_cov_meanfield(xi, x)
        m = compute_mean_meanfield(m, s, xi, x, y)
        xi = compute_xi(x, s, m)
    return m, s


def compute_mean(m0, s0, s, x, y):
    """Compute the updated mean vector."""
    y_adjusted = (y - 0.5).reshape(-1, 1)
    temp2 = np.sum(x * y_adjusted, axis=0).reshape(-1, 1)
    m = s @ (np.linalg.inv(s0) @ m0.reshape(-1, 1) + temp2)
    return m.flatten()


def compute_lambda(xi):
    temp = expit(xi)-0.5
    for i in range(xi.shape[0]):
        temp[i] = temp[i]/(2*xi[i] + 1e-5)
    return temp


def compute_xi(x, s, m):
    """Compute the xi vector for adjusting lambda."""
    m = m.reshape(-1, 1)
    temp1 = s + m @ m.T
    xi = np.sqrt(np.sum(x @ temp1 * x, axis=1))
    return xi


def compute_cov(s0, xi, x):
    temp = 0
    lamb = compute_lambda(xi)
    for i in range(x.shape[0]):
        temp = temp+lamb[i]*np.matmul(x[i].reshape(-1,1), x[i].reshape(-1,1).transpose())
    s = np.linalg.inv(s0) + 2 * temp
    return np.linalg.inv(s)


def variational_logistic(x, y):
    """Perform variational logistic regression."""
    max_iter = 100
    xi = np.ones(x.shape[0])
    m0 = np.zeros(x.shape[1])
    s0 = np.eye(x.shape[1])

    for _ in range(max_iter):
        s = compute_cov(s0, xi, x)
        m = compute_mean(m0, s0, s, x, y)
        xi = compute_xi(x, s, m)
    return m, s

def compute_cov_meanfield(xi, x):
    """Compute covariance matrix for the mean-field approximation."""
    lamb = compute_lambda(xi)[:, np.newaxis]
    # Utilize broadcasting instead of np.repeat for efficiency
    weighted_x_square = np.sum(x**2 * lamb, axis=0)
    prec = 1 / (2 * (weighted_x_square + 0.5))
    s = np.diag(prec)  # Create a diagonal matrix directly
    return s


def compute_mean_meanfield(m, s, xi, x, y):
    """Compute mean vector for the mean-field approximation."""
    y = np.repeat(y, x.shape[1], axis=1) - 0.5
    first_term = np.sum(np.multiply(x, y), axis=0)
    temp1 = np.multiply(x, np.repeat(m[np.newaxis, :], x.shape[0], axis=0))
    lamb = compute_lambda(xi)
    lamb = lamb[:, np.newaxis]
    lamb = np.repeat(lamb, x.shape[1], axis=1)
    xl = np.multiply(x, lamb)
    for i in range(x.shape[1]):
        temp2 = 0
        for j in range(x.shape[1]):
            if (j != i):
                temp2 += np.sum(np.multiply(temp1[:, j], xl[:, i]))
        second_term = -2 * temp2
        m[i] = (first_term[i] + second_term) * s[i, i]
    return m

if __name__ == '__main__':
    train_filename = "./data/bank-note/train.csv"
    test_filename = "./data/bank-note/test.csv"
    x, y = get_data(train_filename)
    x_test, y_test = get_data(test_filename)
    print("2(a) : ")
    w_map, cov = laplace_approx(x, y)
    (correct, n) = test(w_map, x_test, y_test)
    like = predictive_likelihood(x_test, y_test, w_map, cov)
    print("\nMean : \n", w_map)
    print("\nCovariance : \n", cov)
    print("\nAccuracy", correct / n)
    print("\nPredictive likelihood : ", like)

    print("2(b) : ")
    cov_hess = np.multiply(cov, np.eye(cov.shape[0]))
    like = predictive_likelihood(x_test, y_test, w_map, cov_hess)
    print("\nMean : \n", w_map)
    print("\nCovariance : \n", cov_hess)
    print("\nTest Accuracy", correct / n)
    print("\nPredictive likelihood : ", like)

    print("2(c) : ")
    w_map_var, cov_var = variational_logistic(x, y)
    w_map_var = w_map_var.squeeze()
    (correct, n) = test(w_map_var, x_test, y_test)
    like = predictive_likelihood(x_test, y_test, w_map_var, cov_var)
    print("\nMean : \n", w_map_var)
    print("\nCovariance : \n", cov_var)
    print("\nTest Accuracy", correct / n)
    print("\nPredictive likelihood : ", like)

    print("2(d) : ")
    w_map_varm, cov_varm = variational_logistic_meanfield(x, y)
    (correct, n) = test(w_map_varm, x_test, y_test)
    like = predictive_likelihood(x_test, y_test, w_map_varm, cov_varm)
    print("\nMean : \n", w_map_varm)
    print("\nCovariance : \n", cov_varm)
    print("\nTest Accuracy", correct / n)
    print("\nPredictive likelihood : ", like)