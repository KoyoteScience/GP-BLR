from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from os.path import expanduser


# from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html
def plot_prediction_uncertainty_and_data(x, f, X, y, y_pred, sigma=None, ylim=None, xlim=None, title='', filename='regression_results.png'):
    plt.clf()
    plt.figure()
    plt.plot(x, f(x), 'r:', label=r'$f(x) = objective$')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    if isinstance(y_pred, (tuple, list, np.ndarray)) and isinstance(y_pred[0], (tuple, list, np.ndarray)) and len(y_pred[0]) > 1:
        for row_index, y_pred_row in enumerate(y_pred):
            plt.plot(x, y_pred_row, 'b-', label='Prediction' if row_index == 0 else None)
    else:
        plt.plot(x, y_pred, 'b-', label='Prediction')
    if sigma is not None:
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
    plt.savefig(expanduser('~/Downloads/' + filename), dpi=300)
    plt.show()


def gaussian(X1, X2, widths=None):
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-0.5 / np.array(widths) ** 2 * sqdist)


def dot_product(X1, X2):
    return np.outer(X1, X2)


def kernel(X1, X2=None, widths=None, noise_parameter=0.0, mean=0.0, add_constant=False, _normalize=True, multiplier=1):
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

    core_kernel = gaussian(X1, X2, widths=widths)

    if self_kernel:
        white_noise = np.eye(len(core_kernel)) * noise_parameter
        constant = (np.ones(core_kernel.shape)) * mean
    else:
        white_noise = np.zeros(core_kernel.shape)
        constant = np.ones(core_kernel.shape) * mean

    unnormalized_kernel = core_kernel + white_noise ** 2 + constant

    if _normalize:
        normalized_kernel = (unnormalized_kernel.T / np.sqrt(np.diag(
            kernel(X1, widths=widths, noise_parameter=0, mean=0, add_constant=add_constant, _normalize=False)))).T
        normalized_kernel = normalized_kernel / np.sqrt(
            np.diag(kernel(X2, widths=widths, noise_parameter=0, mean=0, add_constant=add_constant, _normalize=False)))

        if add_constant:
            return multiplier * np.hstack([np.ones((len(normalized_kernel), 1)), normalized_kernel])
        else:
            return multiplier * normalized_kernel
    else:
        if add_constant:
            return multiplier * np.hstack([np.ones((len(unnormalized_kernel), 1)), unnormalized_kernel])
        else:
            return multiplier * unnormalized_kernel


######
# Settings
######

compare_specialized_modules = False # this requires access to our libraries which aren't yet public
basis_domain = (-5, 5)
plot_domain = (-1, 2)
training_domain = (0, 1)
ylim = (-2, 2)
widths = [0.5]
wavelength = 0.5
phase = 0
vertical_shift = 0
objective_scale = 5
noise_parameter_scale = 0.1
number_of_basis_functions = 100
number_of_evaluation_points = 500
number_of_training_points = 20
fit_intercept = True
constant_kernel_weight = 1e5
plot_range = (-4 * objective_scale + vertical_shift, 4 * objective_scale + vertical_shift)
render_plots = True


#####


def objective(X):
    return objective_scale * np.sin((X / wavelength + phase) * 2 * np.pi) + vertical_shift


def get_average_subsequent_differences_in_list(ini_list):
    diff_list = []
    for x, y in zip(ini_list[0::], ini_list[1::]):
        diff_list.append(y - x)
    return np.average(diff_list)


number_of_dirac_basis_functions = max(1000, number_of_basis_functions)

basis_centers = np.linspace(basis_domain[0], basis_domain[1], number_of_basis_functions).reshape(-1, 1)
dirac_basis_centers = np.linspace(basis_domain[0], basis_domain[1], number_of_dirac_basis_functions).reshape(-1, 1)
dirac_basis_increment = get_average_subsequent_differences_in_list(dirac_basis_centers)
X = np.linspace(plot_domain[0], plot_domain[1], number_of_evaluation_points).reshape(-1, 1)
X_train = np.linspace(training_domain[0], training_domain[1], number_of_training_points).reshape(-1, 1)
Y_train = objective(X_train).reshape(-1, 1)

noise_parameter = max(1e-5, noise_parameter_scale) * objective_scale
prior_noise_parameter = objective_scale

GP_RBF_kernel = partial(kernel, widths=widths, noise_parameter=1e-3, mean=constant_kernel_weight)
GP_kernel = partial(kernel, widths=widths, noise_parameter=noise_parameter / objective_scale,
                    mean=constant_kernel_weight, multiplier=objective_scale ** 2)
incorrect_kernel = partial(kernel, widths=[x / 5 for x in widths], noise_parameter=noise_parameter)
BLR_basis = partial(kernel, X2=basis_centers,
                    widths=[x / np.sqrt(2) for x in widths],
                    mean=0.0,
                    add_constant=fit_intercept)  # note that noise_parameter only applies if X2 is None
dirac_basis = partial(kernel, X2=dirac_basis_centers,
                      widths=[dirac_basis_increment for x in widths],
                      mean=0.0)  # note that noise_parameter only applies if X2 is None


def normalize_basis(basis, apply_sqrt=True, ignore_first_column=False):
    if ignore_first_column:
        basis_for_norm = basis[:, 1:]
        output_basis = basis.copy()
        output_basis[:, 1:] = (basis[:, 1:].T / np.sqrt(np.diag(basis_for_norm @ basis_for_norm.T))).T
        return output_basis
    else:
        return (basis.T / np.sqrt(np.diag(basis @ basis.T))).T


# regular (manual) basis

X_train_basis = BLR_basis(X_train)
X_basis = BLR_basis(X)
normalization_constant = np.average(np.diag(X_train_basis @ X_train_basis.T)) ** -0.5
X_train_basis = normalize_basis(X_train_basis, ignore_first_column=fit_intercept)
X_basis = normalize_basis(X_basis, ignore_first_column=fit_intercept)

# apply eigenbasis
K = GP_RBF_kernel(dirac_basis_centers)
eigenvalues, eigenvectors = np.linalg.eigh(K)
eigenbasis = eigenvectors.T

X_train_dirac_basis = dirac_basis(X_train)
X_dirac_basis = dirac_basis(X)
X_train_dirac_basis = np.square(normalize_basis(X_train_dirac_basis))
X_dirac_basis = np.square(normalize_basis(X_dirac_basis))

X_train_eigenbasis = X_train_dirac_basis @ eigenbasis.T @ np.diag(np.sqrt(eigenvalues))
X_eigenbasis = X_dirac_basis @ eigenbasis.T @ np.diag(np.sqrt(eigenvalues))

eigenvalues, eigenvectors = np.linalg.eigh(noise_parameter ** -2 * X_train_basis.T @ X_train_basis)
number_of_effective_parameters = sum(x / (prior_noise_parameter ** -2 + x) for x in np.real(eigenvalues))
print(f'number of effective parameters (gamma): {number_of_effective_parameters}')
print(f'number of training points: {number_of_training_points}')

######
# BLR
######

if fit_intercept:
    number_of_features_with_intercept = number_of_basis_functions + 1
else:
    number_of_features_with_intercept = number_of_basis_functions
regularization_matrix = noise_parameter ** 2 * prior_noise_parameter ** -2 * np.eye(number_of_features_with_intercept)
if fit_intercept:
    regularization_matrix[0, 0] = 0
S_matrix = np.linalg.pinv(X_train_basis.T @ X_train_basis + regularization_matrix)
beta_means = S_matrix @ X_train_basis.T @ Y_train
BLR_predictions = X_basis @ beta_means
BLR_predictions_at_X_train = X_train_basis @ beta_means
BLR_cov = noise_parameter ** 2 * (X_basis @ S_matrix @ X_basis.T + np.eye(len(X_basis)))
BLR_sigmas = np.sqrt(np.abs(np.diag(BLR_cov))).reshape(-1, 1)
empirical_noise_parameter = np.sqrt(np.sum((X_train_basis @ beta_means - Y_train) ** 2) / len(BLR_predictions))

######
# BLR -- Eigenbasis
######
prior_covariance_eigen_matrix = prior_noise_parameter ** 2 * np.eye(len(X_train_eigenbasis.T))
covariance_eigen_matrix = np.linalg.pinv(
    noise_parameter ** -2 * X_train_eigenbasis.T @ X_train_eigenbasis + np.linalg.pinv(prior_covariance_eigen_matrix))
beta_eigen_means = noise_parameter ** -2 * covariance_eigen_matrix @ X_train_eigenbasis.T @ Y_train
BLR_eigen_predictions = X_eigenbasis @ beta_eigen_means
BLR_eigen_predictions_at_X_train = X_train_eigenbasis @ beta_eigen_means
BLR_eigen_cov = X_eigenbasis @ covariance_eigen_matrix @ X_eigenbasis.T + noise_parameter ** 2 * np.eye(
    len(X_eigenbasis))
BLR_eigen_sigmas = np.sqrt(np.abs(np.diag(BLR_eigen_cov))).reshape(-1, 1)
eigen_empirical_noise_parameter = np.sqrt(
    np.sum((X_train_eigenbasis @ beta_eigen_means - Y_train) ** 2) / len(BLR_eigen_predictions))

######
# BLR - Specialized modules
######

if compare_specialized_modules:
    from src.bayes_linear_regressor import BayesLinearRegressor, BayesLinearAlgebraLinearRegressor
    from src.support_regressors import BasisAdapter, TuningAdapter

    # note that l2_regularization_constant is alpha/beta in http://krasserm.github.io/2019/02/23/bayesian-linear-regression/
    #      that is, with alpha = prior_noise_parameter ** -2 = 1,
    #      l2_regularization_constant = noise_parameter ** 2

    BLR_learn_sigma_and_prior_noise_parameter_object = BasisAdapter(
        regressor=BayesLinearRegressor(
            l2_regularization_constant=None,
            fixed_prior_noise_parameter=None,
            fixed_noise_parameter=None,
            fit_intercept=fit_intercept
        ),
        domains=[basis_domain],
        sampling_factors=number_of_basis_functions,
        widths=[x / np.sqrt(2) for x in widths]
    )  # note that kernel_noise_parameter default is 0
    BLR_learn_sigma_and_prior_noise_parameter_object.fit(X_train, Y_train)
    BLR_learn_sigma_and_prior_noise_parameter_object_predictions, \
    BLR_learn_sigma_and_prior_noise_parameter_object_sigmas = \
        BLR_learn_sigma_and_prior_noise_parameter_object.predict(X, return_std=True)

    BLR_learn_sigma_object = BasisAdapter(
        regressor=BayesLinearRegressor(
            l2_regularization_constant=None,
            fixed_prior_noise_parameter=prior_noise_parameter,
            fixed_noise_parameter=None,
            fit_intercept=fit_intercept
        ),
        domains=[basis_domain],
        sampling_factors=number_of_basis_functions,
        widths=[x / np.sqrt(2) for x in widths]
    )  # note that kernel_noise_parameter default is 0
    BLR_learn_sigma_object.fit(X_train, Y_train)
    BLR_learn_sigma_object_predictions, BLR_learn_sigma_object_sigmas = BLR_learn_sigma_object.predict(X,
                                                                                                       return_std=True)

    BLR_learn_sigma_evidence_object = TuningAdapter(
        regressor=BasisAdapter,
        regressor_keyword_arguments={

            'regressor': BayesLinearRegressor,
            'l2_regularization_constant': None,
            'fixed_prior_noise_parameter': prior_noise_parameter,
            'fixed_noise_parameter': None,
            'fit_intercept': fit_intercept,

            'domains': [basis_domain],
            'sampling_factors': number_of_basis_functions,
            'widths': [x / np.sqrt(2) for x in widths]
        },
        hyperparameter_domains={
            'fixed_noise_parameter': [1e-5, 10]
        }
    )
    BLR_learn_sigma_evidence_object.fit(X_train, Y_train)
    BLR_learn_sigma_evidence_object_predictions, BLR_learn_sigma_evidence_object_sigmas = BLR_learn_sigma_evidence_object.predict(
        X,
        return_std=True)
    

    BLR_learn_sigma_and_width_evidence_object = TuningAdapter(
        regressor=BasisAdapter,
        regressor_keyword_arguments={

            'regressor': BayesLinearRegressor,
            'l2_regularization_constant': None,
            'fixed_prior_noise_parameter': prior_noise_parameter,
            'fixed_noise_parameter': None,
            'fit_intercept': fit_intercept,

            'domains': [basis_domain],
            'sampling_factors': number_of_basis_functions,
            'widths': [x / np.sqrt(2) for x in widths]
        },
        hyperparameter_domains={
            'fixed_noise_parameter': [1e-5, 10],
            'widths': [[1e-2, 10]]
        }
    )
    BLR_learn_sigma_and_width_evidence_object.fit(X_train, Y_train)
    BLR_learn_sigma_and_width_evidence_object_predictions, BLR_learn_sigma_and_width_evidence_object_sigmas = BLR_learn_sigma_and_width_evidence_object.predict(
        X,
        return_std=True)

    BLR_learn_prior_noise_parameter_object = BasisAdapter(
        regressor=BayesLinearRegressor(
            l2_regularization_constant=None,
            fixed_prior_noise_parameter=None,
            fixed_noise_parameter=noise_parameter,
            fit_intercept=fit_intercept
        ),
        domains=[basis_domain],
        sampling_factors=number_of_basis_functions,
        widths=[x / np.sqrt(2) for x in widths]
    )  # note that kernel_noise_parameter default is 0
    BLR_learn_prior_noise_parameter_object.fit(X_train, Y_train)
    BLR_learn_prior_noise_parameter_object_predictions, BLR_learn_prior_noise_parameter_object_sigmas = BLR_learn_prior_noise_parameter_object.predict(
        X, return_std=True)

    BLR_fixed_regularization_learn_sigma_object = BasisAdapter(
        regressor=BayesLinearRegressor(
            l2_regularization_constant=1e-1,  # noise_parameter**2 / prior_noise_parameter**2,
            fixed_prior_noise_parameter=None,
            fixed_noise_parameter=None,
            fit_intercept=fit_intercept,
            use_empirical_noise_parameter=False
        ),
        domains=[basis_domain],
        sampling_factors=number_of_basis_functions,
        widths=[x / np.sqrt(2) for x in widths]
    )  # note that kernel_noise_parameter default is 0
    BLR_fixed_regularization_learn_sigma_object.fit(X_train, Y_train)
    BLR_fixed_regularization_learn_sigma_object_predictions, \
    BLR_fixed_regularization_learn_sigma_object_sigmas = \
        BLR_fixed_regularization_learn_sigma_object.predict(X, return_std=True)

    BLR_fixed_prior_noise_parameter_object = BasisAdapter(
        regressor=BayesLinearRegressor(
            l2_regularization_constant=None,
            fixed_prior_noise_parameter=prior_noise_parameter,
            fixed_noise_parameter=noise_parameter,
            fit_intercept=fit_intercept
        ),
        domains=[basis_domain],
        sampling_factors=number_of_basis_functions,
        widths=[x / np.sqrt(2) for x in widths]
    )  # note that kernel_noise_parameter default is 0
    BLR_fixed_prior_noise_parameter_object.fit(X_train, Y_train)
    BLR_fixed_prior_noise_parameter_object_predictions, BLR_fixed_prior_noise_parameter_object_sigmas = BLR_fixed_prior_noise_parameter_object.predict(
        X, return_std=True)

    BLR_object = BasisAdapter(
        regressor=BayesLinearRegressor(
            l2_regularization_constant=noise_parameter ** 2 / prior_noise_parameter ** 2,
            fixed_prior_noise_parameter=None,
            fixed_noise_parameter=noise_parameter,
            fit_intercept=fit_intercept
        ),
        domains=[basis_domain],
        sampling_factors=number_of_basis_functions,
        widths=[x / np.sqrt(2) for x in widths]
    )  # note that kernel_noise_parameter default is 0
    BLR_object_prior_predictions, BLR_object_prior_sigmas = BLR_object.predict(X, return_std=True)
    BLR_object_prior_sample_y = [BLR_object.sample_y(X) for _ in range(10)]
    BLR_object.fit(X_train, Y_train)
    BLR_object_predictions, BLR_object_sigmas = BLR_object.predict(X, return_std=True)
    BLR_object_posterior_sample_y = [BLR_object.sample_y(X) for _ in range(10)]

    BLALR_object = BasisAdapter(
        regressor=BayesLinearAlgebraLinearRegressor(
            l2_regularization_constant=noise_parameter ** 2 / prior_noise_parameter ** 2,
            fixed_prior_noise_parameter=None,
            fixed_noise_parameter=noise_parameter,
            fit_intercept=fit_intercept
        ),
        domains=[basis_domain],
        sampling_factors=number_of_basis_functions,
        widths=[x / np.sqrt(2) for x in widths]
    )  # note that kernel_noise_parameter default is 0
    BLALR_object.fit(X_train, Y_train)
    BLALR_object_predictions, BLALR_object_sigmas = BLALR_object.predict(X, return_std=True)

    # np.sum(np.abs(X_train_basis - BLR_object.transform(X_train))) / X_train_basis.size
    # np.sum(np.abs(X_basis - BLR_object.transform(X))) / X_basis.size

    train_index = round(X_train_basis.shape[0] / 2)
    basis_index = round(X_train_basis.shape[1] / 2)

    if render_plots:
        plt.clf()
        plt.figure()
        y = BLR_object.transform(X_train[train_index]).T
        print(np.max(y))
        plt.plot(list(range(len(y))), y, 'r')
        plt.show()

        plt.clf()
        plt.figure()
        y = X_train_basis[train_index]
        print(np.max(y))
        plt.plot(list(range(len(y))), y, 'b')
        plt.show()

######
# GP
######

K = GP_kernel(X_train)
K_s = GP_kernel(X_train, X)
K_ss = GP_kernel(X)
K_inv = np.linalg.pinv(K)
GP_predictions = K_s.T.dot(K_inv).dot(Y_train)
GP_cov = (K_ss - K_s.T.dot(K_inv).dot(K_s))
GP_sigmas = np.sqrt(np.abs(np.diag(GP_cov))).reshape(-1, 1)

######
# Plots
######

if render_plots:
    plot_prediction_uncertainty_and_data(
        X, objective, X_train, Y_train, GP_predictions, GP_sigmas, plot_range, plot_domain,
        'Gaussian Process'
    )

    plot_prediction_uncertainty_and_data(
        X, objective, X_train, Y_train, BLR_predictions, BLR_sigmas, plot_range, plot_domain,
        'Bayesian Linear Regression'
    )

    plot_prediction_uncertainty_and_data(
        X, objective, X_train, Y_train, BLR_eigen_predictions, BLR_eigen_sigmas, plot_range, plot_domain,
        'Bayesian Linear Regression (Eigenbasis)'
    )

    if compare_specialized_modules:
        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_object_prior_predictions, BLR_object_prior_sigmas,
            plot_range,
            plot_domain,
            'Bayesian Linear Regression Prior',
            'prior_regression_results.png'
        )
        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_object_prior_sample_y, None,
            plot_range,
            plot_domain,
            'Bayesian Linear Regression Prior Samples',
            'prior_samples.png'
        )
        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_object_posterior_sample_y, None,
            plot_range,
            plot_domain,
            'Bayesian Linear Regression Posterior Samples',
            'posterior_samples.png'
        )
        
        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_learn_sigma_object_predictions, BLR_learn_sigma_object_sigmas,
            plot_range,
            plot_domain,
            'Bayesian Linear Regression with Learned $\sigma$ and Fixed $\sigma_p$'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_learn_sigma_evidence_object_predictions, BLR_learn_sigma_evidence_object_sigmas,
            plot_range,
            plot_domain,
            'Bayesian Linear Regression with Learned $\sigma$ and Fixed $\sigma_p$ with Evidence'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_learn_sigma_and_width_evidence_object_predictions, BLR_learn_sigma_and_width_evidence_object_sigmas,
            plot_range,
            plot_domain,
            'Bayesian Linear Regression with Learned $\sigma$ and $l$, Fixed $\sigma_p$ with Evidence'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_learn_prior_noise_parameter_object_predictions,
            BLR_learn_prior_noise_parameter_object_sigmas, plot_range, plot_domain,
            'Bayesian Linear Regression with Fixed $\sigma$ and Learned $\sigma_p$'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_learn_sigma_and_prior_noise_parameter_object_predictions,
            BLR_learn_sigma_and_prior_noise_parameter_object_sigmas, plot_range, plot_domain,
            'Bayesian Linear Regression with Learned $\sigma$ and $\sigma_p$'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_fixed_regularization_learn_sigma_object_predictions,
            BLR_fixed_regularization_learn_sigma_object_sigmas, plot_range, plot_domain,
            'Bayesian Linear Regression with Learned $\sigma$ and Fixed $\lambda$'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_fixed_prior_noise_parameter_object_predictions,
            BLR_fixed_prior_noise_parameter_object_sigmas, plot_range, plot_domain,
            'Bayesian Linear Regression with Fixed $\sigma_p$ and $\sigma$'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLR_object_predictions, BLR_object_sigmas, plot_range, plot_domain,
            'Bayesian Linear Regression with Fixed $\lambda=\sigma^2/\sigma_p^2$'
        )

        plot_prediction_uncertainty_and_data(
            X, objective, X_train, Y_train, BLALR_object_predictions, BLALR_object_sigmas, plot_range, plot_domain,
            'Bayesian Linear Regression with Fixed $\lambda=\sigma^2/\sigma_p^2$ (alt.)'
        )

######
# Eigendecomposition
######

if render_plots:

    K = GP_kernel(basis_centers)
    eigenvalues, eigenvectors = np.linalg.eig(K)
    comparison = GP_kernel(basis_centers)

    plt.clf()
    plt.figure()
    plt.plot(basis_centers, eigenvectors.T[0], 'red', label='eigenbasis[0]')
    plt.plot(basis_centers, eigenvectors.T[1], 'green', label='eigenbasis[1]')
    plt.plot(basis_centers, eigenvectors.T[2], 'blue', label='eigenbasis[2]')
    plt.plot(basis_centers, eigenvectors.T[3], 'orange', label='eigenbasis[3]')
    plt.xlim(plot_domain)
    plt.legend(loc='lower right')
    plt.title('Bayesian Linear Regression Eigenbasis Examples')
    plt.savefig(expanduser('~/Downloads/eigenbasis.png'), dpi=300)
    plt.show()

    basis = BLR_basis(basis_centers)

    index_increment = round(len(basis) * 0.1)
    plt.clf()
    plt.figure()
    if fit_intercept:
        plt.plot(basis_centers, basis[0, 1:], 'red', label='basis 0%')
        plt.plot(basis_centers, basis[index_increment, 1:], 'green', label='basis 10%')
        plt.plot(basis_centers, basis[2 * index_increment, 1:], 'blue', label='basis 20%')
        plt.plot(basis_centers, basis[3 * index_increment, 1:], 'orange', label='basis 30%')
    else:
        plt.plot(basis_centers, basis[0], 'red', label='basis 0%')
        plt.plot(basis_centers, basis[index_increment], 'green', label='basis 10%')
        plt.plot(basis_centers, basis[2 * index_increment], 'blue', label='basis 20%')
        plt.plot(basis_centers, basis[3 * index_increment], 'orange', label='basis 30%')

    plt.xlim(plot_domain)
    plt.legend(loc='upper right')
    plt.title('Bayesian Linear Regression Manual Basis Examples')
    plt.savefig(expanduser('~/Downloads/manualbasis.png'), dpi=300)
    plt.show()

######
# Log-likelihood Comparison
######

K = GP_kernel(X_train)
K_inv = np.linalg.pinv(K)
if fit_intercept:
    beta_means_for_log_likelihood = beta_means[1:]
else:
    beta_means_for_log_likelihood = beta_means

log_likelihood_first_term_GP = float((Y_train.T - np.average(Y_train)) @ K_inv @ (Y_train - np.average(Y_train)))
log_likelihood_first_term_BLR = noise_parameter ** -2 * np.sum((Y_train - BLR_predictions_at_X_train) ** 2, axis=None) + \
                                prior_noise_parameter ** -2 * beta_means_for_log_likelihood.T @ beta_means_for_log_likelihood
log_likelihood_first_term_BLR_eigen = noise_parameter ** -2 * np.sum((Y_train - BLR_eigen_predictions_at_X_train) ** 2,
                                                                     axis=None) + \
                                      prior_noise_parameter ** -2 * beta_means_for_log_likelihood.T @ beta_means_for_log_likelihood

print(f'First term GP: {log_likelihood_first_term_GP}')
print(f'First term BLR: {float(log_likelihood_first_term_BLR)}')
print(f'First term BLR eigenstates: {float(log_likelihood_first_term_BLR_eigen)}')

test_GP_kernel = GP_kernel
test_K = test_GP_kernel(X_train)

inv_prior_covariance_matrix = prior_noise_parameter ** -2 * np.eye(len(X_train_basis.T))
if fit_intercept:
    inv_prior_covariance_matrix[0, 0] = 0.0
covariance_matrix = \
    np.linalg.pinv((noise_parameter ** -2 * X_train_basis.T @ X_train_basis + inv_prior_covariance_matrix))

log_likelihood_second_term_GP = np.log(np.linalg.det(K - constant_kernel_weight))
log_likelihood_second_term_BLR = - np.log(np.linalg.det(covariance_matrix)) + 2 * number_of_training_points * np.log(
    noise_parameter) + 2 * (number_of_basis_functions) * np.log(prior_noise_parameter)
print(f'Second term GP: {log_likelihood_second_term_GP}')
print(f'Second term BLR: {log_likelihood_second_term_BLR}')
print(
    'Note: second term only exactly matches when number_of_basis_functions >> 10, fit_intercept==False, and vertical_shift==0 ')

print(f'Objective: {objective_scale}')
print(f'Empirical sigma: {np.std(Y_train)}')
print(f'Gamma: {number_of_effective_parameters}')
print(f'Prior sigma with fixed sigma: {BLR_learn_prior_noise_parameter_object.regressor.prior_noise_parameter}')
print(f' --> sigma for the same model: {BLR_learn_prior_noise_parameter_object.regressor.noise_parameter}')
print(
    f' --> gamma for the same model: {BLR_learn_prior_noise_parameter_object.regressor.effective_number_of_parameters}')
print(
    f'Prior sigma with learned sigma: {BLR_learn_sigma_and_prior_noise_parameter_object.regressor.prior_noise_parameter}')
print(f' --> sigma for the same model: {BLR_learn_sigma_and_prior_noise_parameter_object.regressor.noise_parameter}')
print(
    f' --> gamma for the same model: {BLR_learn_sigma_and_prior_noise_parameter_object.regressor.effective_number_of_parameters}')
