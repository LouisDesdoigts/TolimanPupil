import jax.numpy as np
import jax.random as jr
from jax import config
import dLux as dl
from dLux.models import Toliman, SimpleToliman, AlphaCen

if config['jax_enable_x64']:
    basis_path = "data/basis_64.npy"
else:
    basis_path = "data/basis_32.npy"

def get_dynamic_mask():
    basis = np.load(basis_path)
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