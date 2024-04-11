import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def read_data(filename):
    """Reads data from a file."""
    with open(filename, 'r') as f:
        x = np.array([[float(num) for num in line.split()] for line in f])
    return x


def norm_data(x):
    """Normalizes the data within [-1, 1] and plots it."""
    normalized_x = (x - np.mean(x, axis=0)) / (np.ptp(x, axis=0))
    return normalized_x


def plot_data(x, cluster_id, k, mean, iter_num):
    """Plots data with cluster identifiers."""
    colors = ['r', 'b']
    plt.clf()
    for i in range(k):
        plt.scatter(x[cluster_id == i, 0], x[cluster_id == i, 1], color=colors[i])
    plt.scatter(mean[:, 0], mean[:, 1], color='g', marker='D')
    plt.savefig(f"iter_{iter_num}.png")
    plt.show()


def gmm_init(k, initial_type):
    """Initializes GMM parameters."""
    if initial_type == 1:
        mean = np.array([[-1, 1], [1, -1]])
        covariance = np.tile(0.1 * np.eye(2), (k, 1, 1))
        mix = np.full(k, 1 / k)
    else:
        mean = np.array([[-1, -1], [1, 1]])
        covariance = np.tile(0.5 * np.eye(2), (k, 1, 1))
        mix = np.full(k, 1 / k)
    print(f"Initialization done.\nMean = {mean}\nCovariance = {covariance}\nMixing Coefficients = {mix}")
    return mean, covariance, mix


def e_step(x, k, mean, covariance, mix):
    """Performs the E-step of the GMM algorithm."""
    gamma = np.array([mix[i] * multivariate_normal.pdf(x, mean=mean[i], cov=covariance[i]) for i in range(k)]).T
    return gamma / gamma.sum(axis=1)[:, np.newaxis]


def m_step(x, k, gamma):
    """Performs the M-step of the GMM algorithm."""
    n_k = gamma.sum(axis=0)
    mix = n_k / x.shape[0]
    mean = np.dot(gamma.T, x) / n_k[:, np.newaxis]
    covariance = np.array([
        np.dot((gamma[:, i:i + 1] * (x - mean[i])).T, x - mean[i]) / n_k[i]
        for i in range(k)
    ])
    return mean, covariance, mix


def gmm(k, filename, initial_type):
    """Executes the Gaussian Mixture Model algorithm."""
    x = norm_data(read_data(filename))
    mean, covariance, mix = gmm_init(k, initial_type)
    plot_iter = [1, 2, 5, 100]

    for i in range(1, max_iter + 1):
        gamma = e_step(x, k, mean, covariance, mix)
        mean, covariance, mix = m_step(x, k, gamma)

        if i in plot_iter:
            cluster_id = np.argmax(gamma, axis=1)
            plot_data(x, cluster_id, k, mean, i)


    print(f"Final cluster assignments for {x.shape[0]} data points available.")

if __name__ == '__main__':
    max_iter = 100
    initial_type = 1
    gmm(2, "./data/faithful/faithful.txt", initial_type)
    initial_type = 2
    gmm(2, "./data/faithful/faithful.txt", initial_type)
