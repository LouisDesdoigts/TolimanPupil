import jax.numpy as np
import jax.random as jr
import zodiax as zdx
import optax
from models import point_model, binary_model

####################################
### Gradient Energy Optimisation ###
####################################
from gradient_energy import GE, RGE, pix_per_fringe, radial_mask

# Gradient Energy Loss function
@zdx.filter_jit
@zdx.filter_value_and_grad('mask.coefficients')
def GE_loss_fn(model, rmin=0, rmax=10, power=0.5):
    # Get PSF
    psf = model.model()
    ppf = pix_per_fringe(model)
    mask = radial_mask(psf.shape[0], rmin * ppf, rmax * ppf)

    # Calcualte loss
    loss1 = - np.power(mask*GE(psf),   power).sum()
    loss2 = - np.power(mask*RGE(psf),  power).sum()
    return loss1 + loss2


# Optimisation fn
def optimise_ge(seed, rmin=0, rmax=10, power=0.5, lr=1e0, epochs=100):

    model = point_model()

    # Initialise model
    shape = model.mask.coefficients.shape
    model = model.set('mask.coefficients', jr.normal(jr.PRNGKey(seed), shape))
    optimiser, state = zdx.get_optimiser(model, 'mask.coefficients', 
        optax.adam(lr))

    # Optimisation loop
    losses, coefficients = [], [model.mask.coefficients]
    for i in range(epochs):
        loss, grads = GE_loss_fn(model, rmin=rmin, rmax=rmax, power=power)
        step, state = optimiser.update(grads, state)
        model = zdx.apply_updates(model, step)
        losses.append(loss)
        coefficients.append(model.mask.coefficients)
    
    return losses, coefficients


#######################################
### Fisher Information Optimisation ###
#######################################
from bayes import get_covaraince

# Loss function
@zdx.filter_jit
@zdx.filter_value_and_grad('mask.coefficients')
def fisher_loss_fn(model, parameters):
    covariance_matrix = get_covaraince(model, parameters)
    return covariance_matrix[0, 0] ** 0.5


# Optimisation fn
def optimise_fisher(seed, parameters, lr=1e0, epochs=100):

    model = binary_model()

    # Initialise model
    shape = model.mask.coefficients.shape
    model = model.set('mask.coefficients', jr.normal(jr.PRNGKey(seed), shape))
    optimiser, state = zdx.get_optimiser(model, 'mask.coefficients', 
        optax.adam(lr))

    # Optimisation loop
    losses, coefficients = [], [model.mask.coefficients]
    for i in range(epochs):
        loss, grads = fisher_loss_fn(model, parameters)
        step, state = optimiser.update(grads, state)
        model = zdx.apply_updates(model, step)
        losses.append(loss)
        coefficients.append(model.mask.coefficients)
    
    return losses, coefficients