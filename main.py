from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from os.path import expanduser


# from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
def plot_prediction_uncertainty_and_data(x, f, X, y, y_pred, sigma, ylim=None, xlim=None, title=''):
    plt.clf()
    plt.figure()
    plt.plot(x, f(x), 'r:', label=r'$f(x) = objective$')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(expanduser('~/Downloads/regression_results.png'), dpi=300)
    plt.show()


def gaussian(X1, X2, sigma_f=1.0):
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / np.array(widths) ** 2 * sqdist)


def squared_gaussian(X1, X2):
    gaussian_values = gaussian(X1, X2)
    return np.square(gaussian_values)


def sqrt_gaussian(X1, X2):
    gaussian_values = gaussian(X1, X2)
    return np.sqrt(gaussian_values)


def dot_product(X1, X2):
    return np.outer(X1, X2)


def kernel(X1, X2=None, widths=None, sigma_f=1.0, noise_parameter=0.0, core_kernel=gaussian):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """

    if X2 is None:
        self_kernel = True
        X2 = X1
    else:
        self_kernel = False
        X2 = X2

    if widths is None:
        widths = self.widths

    core_kernel = core_kernel(X1, X2)
    white_noise = np.zeros(core_kernel.shape)

    if self_kernel:
        white_noise = np.eye(len(core_kernel)) * noise_parameter

    return core_kernel + white_noise ** 2


######
# Settings
######

basis_domain = (-2, 3)
plot_domain = (-1, 2)
training_domain = (0, 1)
ylim = (-2, 2)
widths = [0.5]
wavelength = 0.2
noise_parameter = 0.2
number_of_basis_functions = 1000
number_of_evaluation_points = 1000
number_of_training_points = 50
apply_basis = True
GP_core_kernel = gaussian
BLR_core_kernel = squared_gaussian
cutoff_eigenvectors = False


#####

def objective(X, wavelength=wavelength):
    return np.sin(X * 2 * np.pi / wavelength)


basis_centers = np.linspace(basis_domain[0], basis_domain[1], number_of_basis_functions).reshape(-1, 1)
X = np.linspace(plot_domain[0], plot_domain[1], number_of_evaluation_points).reshape(-1, 1)
X_train = np.linspace(training_domain[0], training_domain[1], number_of_training_points).reshape(-1, 1)
Y_train = objective(X_train).reshape(-1, 1)

GP_kernel = partial(kernel, widths=widths, core_kernel=GP_core_kernel, noise_parameter=noise_parameter)
BLR_basis = partial(kernel, X2=basis_centers, widths=widths, core_kernel=BLR_core_kernel,
                    noise_parameter=noise_parameter)
dirac_basis = partial(kernel, X2=basis_centers, widths=0.001, core_kernel=BLR_core_kernel, noise_parameter=0.0)

if apply_basis:

    # regular (manual) basis
    X_train_basis = BLR_basis(X_train)
    X_basis = BLR_basis(X)
    normalization_constant = np.average(np.diag(X_train_basis @ X_train_basis.T)) ** -0.5
    X_train_basis *= normalization_constant
    X_basis *= normalization_constant
    prior_covariance_scale = noise_parameter ** -2

    # apply eigenbasis
    K = GP_kernel(basis_centers)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    sorted_indices = sorted(list(range(len(eigenvalues))), key=lambda x: -np.real(eigenvalues[x]))
    eigenvalues = [eigenvalues[i] for i in sorted_indices]
    eigenvectors = np.array([eigenvectors[i] for i in sorted_indices])
    eigenvectors_transposed = np.array([eigenvectors.T[i] for i in sorted_indices])
    eigenvectors = eigenvectors_transposed.T

    # remove low-eigenvalue eigenvectors
    if cutoff_eigenvectors:
        cutoff = noise_parameter ** 2
        use_eigenindices = [i for i, val in enumerate(eigenvalues) if val > cutoff] 
        print(f'Kep {len(use_eigenindices)} out of {len(eigenvalues)} eigenvectors')
        eigenvectors = np.array([eigenvectors[i] for i in use_eigenindices])
    
    
    BLR_eigenbasis = eigenvectors.T
    X_train_dirac_basis = dirac_basis(X_train)
    X_dirac_basis = dirac_basis(X)
    normalization_constant = np.average(np.diag(X_train_dirac_basis @ X_train_dirac_basis.T)) ** -0.5
    X_train_dirac_basis *= normalization_constant
    X_dirac_basis *= normalization_constant
    X_train_eigenbasis = X_train_dirac_basis @ BLR_eigenbasis
    X_eigenbasis = X_dirac_basis @ BLR_eigenbasis
    normalization_constant = np.average(np.diag(X_train_eigenbasis @ X_train_eigenbasis.T)) ** -0.5
    X_train_eigenbasis *= normalization_constant
    X_eigenbasis *= normalization_constant

else:
    X_train_basis = X_train
    X_basis = X
    if BLR_prior_covariance_scale is None:
        prior_covariance_scale = 1e8

    X_train_eigenbasis = X_train
    X_eigenbasis = X

######
# BLR
######

prior_covariance_matrix = prior_covariance_scale * np.eye(len(X_train_basis.T))
covariance_matrix = np.linalg.pinv(X_train_basis.T @ X_train_basis + np.linalg.pinv(prior_covariance_matrix))
beta_means = covariance_matrix @ X_train_basis.T @ Y_train
BLR_predictions = X_basis @ beta_means
BLR_predictions_at_X_train = X_train_basis @ beta_means
BLR_cov = noise_parameter ** 2 * X_basis @ covariance_matrix @ X_basis.T + noise_parameter ** 2 * np.eye(len(X_basis))
BLR_sigmas = np.sqrt(np.abs(np.diag(BLR_cov))).reshape(-1, 1)
empirical_noise_parameter = np.sqrt(np.sum((X_train_basis @ beta_means - Y_train) ** 2) / len(BLR_predictions))

######
# BLR -- Eigenbasis
######

prior_covariance_matrix = prior_covariance_scale * np.eye(len(X_train_eigenbasis.T))
covariance_matrix = np.linalg.pinv(X_train_eigenbasis.T @ X_train_eigenbasis + np.linalg.pinv(prior_covariance_matrix))
beta_eigen_means = covariance_matrix @ X_train_eigenbasis.T @ Y_train
BLR_eigen_predictions = X_eigenbasis @ beta_eigen_means
BLR_eigen_predictions_at_X_train = X_train_eigenbasis @ beta_eigen_means
BLR_eigen_cov = noise_parameter ** 2 * X_eigenbasis @ covariance_matrix @ X_eigenbasis.T + noise_parameter ** 2 * np.eye(
    len(X_eigenbasis))
BLR_eigen_sigmas = np.sqrt(np.abs(np.diag(BLR_eigen_cov))).reshape(-1, 1)
eigen_empirical_noise_parameter = np.sqrt(
    np.sum((X_train_eigenbasis @ beta_eigen_means - Y_train) ** 2) / len(BLR_eigen_predictions))

######
# BLR_Class
######

# TODO: Write this

######
# GP
######

K = GP_kernel(X_train)
K_s = GP_kernel(X_train, X)
K_ss = GP_kernel(X)
K_inv = np.linalg.pinv(K)
GP_predictions = K_s.T.dot(K_inv).dot(Y_train)
GP_cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
GP_sigmas = np.sqrt(np.abs(np.diag(GP_cov))).reshape(-1, 1)

######
# Plots
######

plot_prediction_uncertainty_and_data(
    X, objective, X_train, Y_train, GP_predictions, GP_sigmas, ylim, plot_domain,
    'Gaussian Process'
)

plot_prediction_uncertainty_and_data(
    X, objective, X_train, Y_train, BLR_predictions, BLR_sigmas, ylim, plot_domain,
    'Bayesian Linear Regression'
)

plot_prediction_uncertainty_and_data(
    X, objective, X_train, Y_train, BLR_eigen_predictions, BLR_eigen_sigmas, ylim, plot_domain,
    'Bayesian Linear Regression (Eigenbasis)'
)

######
# Eigendecomposition
######

K = GP_kernel(basis_centers)
eigenvalues, eigenvectors = np.linalg.eig(K)
comparison = GP_kernel(basis_centers)

plt.clf()
plt.figure()
plt.plot(basis_centers, eigenvectors.T[0], 'red', label='eigenbasis[0]')
plt.plot(basis_centers, eigenvectors.T[1], 'green', label='eigenbasis[1]')
plt.plot(basis_centers, eigenvectors.T[2], 'blue', label='eigenbasis[2]')
plt.plot(basis_centers, eigenvectors.T[3], 'orange', label='eigenbasis[3]')
plt.legend(loc='lower right')
plt.title('Bayesian Linear Regression Eigenbasis Examples')
plt.savefig(expanduser('~/Downloads/eigenbasis.png'), dpi=300)
plt.show()

basis = BLR_basis(basis_centers)

index_increment = round(len(basis) * 0.1)
plt.clf()
plt.figure()
plt.plot(basis_centers, basis[0], 'red', label='basis 0%')
plt.plot(basis_centers, basis[index_increment], 'green', label='basis 10%')
plt.plot(basis_centers, basis[2*index_increment], 'blue', label='basis 20%')
plt.plot(basis_centers, basis[3*index_increment], 'orange', label='basis 30%')
plt.legend(loc='upper right')
plt.title('Bayesian Linear Regression Manual Basis Examples')
plt.savefig(expanduser('~/Downloads/manualbasis.png'), dpi=300)
plt.show()

######
# Log-likelihood Comparison
######

K = GP_kernel(X_train)
K_inv = np.linalg.pinv(K)

log_likelihood_first_term_GP = float(Y_train.T @ K_inv @ Y_train)
log_likelihood_first_term_BLR = noise_parameter ** -2 * np.sum((Y_train - BLR_predictions_at_X_train)**2,axis=None)
log_likelihood_first_term_BLR_eigen = noise_parameter ** -2 * np.sum((Y_train - BLR_eigen_predictions_at_X_train)**2,axis=None)

log_likelihood_second_term_GP = np.linalg.det(K)
log_likelihood_second_term_BLR = noise_parameter ** -2 * beta_means.T @ beta_means
log_likelihood_second_term_BLR_eigen = noise_parameter ** -2 * beta_eigen_means.T @ beta_eigen_means


