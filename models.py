import jax.numpy as np
import jax.random as jr
import dLux as dl
from dLux.models import Toliman, SimpleToliman, AlphaCen

def get_dynamic_mask():
    basis = np.load("data/basis.npy")
    mask_coeffs = jr.normal(jr.PRNGKey(0), (len(basis),))
    mask = dl.ApplyBasisCLIMB(basis, 595e-9, mask_coeffs)
    return mask

def point_model(zernikes=np.arange(4, 11)):
    mask = get_dynamic_mask()
    optics = SimpleToliman(mask=mask, zernikes=zernikes, psf_npixels=128)
    source = dl.PointSource(wavelengths=AlphaCen().wavelengths)
    model = Toliman(optics, source)
    return model

def binary_model(zernikes=np.arange(4, 11)):
    mask = get_dynamic_mask()
    optics = SimpleToliman(mask=mask, zernikes=zernikes)
    source = AlphaCen(log_flux=0)
    model = Toliman(optics, source)
    return model