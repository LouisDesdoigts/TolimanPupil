import jax.numpy as np
import jax.scipy as jsp
from jax import hessian

# Perturbation fn
def perturb_fn(X, model, parameters):
    """Perturbs the model parameters by the values in X."""
    if parameters[-1] == 'aberrations.coefficients':
        params = parameters[:-1]
    else:
        params = parameters
    
    for parameter, x in zip(params, X):
        model = model.add(parameter, x)
    
    if parameters[-1] == 'aberrations.coefficients':
        nzernikes = len(model.aberrations.coefficients)
        model = model.add('aberrations.coefficients', X[-nzernikes:])
    return model


def likelihood(X, model, data, perturb_fn, parameters):
    """
    Calculates the log likelihood of the model after perturbation given the 
    data.
    """
    psf = perturb_fn(X, model, parameters).model()
    return jsp.stats.poisson.logpmf(data, psf).sum()


def calculate_covariance(X, model, data, perturb_fn, parameters):
    """Calculates the covariance matrix of the model parameters."""
    return -np.linalg.inv(hessian(likelihood)(X, model, data, perturb_fn, 
        parameters))


def get_covaraince(model, parameters, log_flux=6):
    """
    Get the covariance matrix of the model parameters with the given log_flux.
    """
    # Setup
    X = [0.] * len(parameters)
    if "aberrations.coefficients" in parameters:
        X += [0.] * (len(model.aberrations.coefficients) - 1)
    X = np.array(X)

    # Calculate
    model = model.set('log_flux', log_flux)
    return calculate_covariance(X, model, model.model(), perturb_fn, parameters)