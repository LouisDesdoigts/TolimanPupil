import jax.numpy as np
import dLux.utils as dlu


def GE(array):
    """Calcuates the spatial gradient energy of the array."""
    grads_vec = np.gradient(array)
    return np.hypot(grads_vec[0], grads_vec[1])


def RGE(array, epsilon = 1e-8):
    """Calcuates the spatial radial gradient energy of the array."""
    npix = array.shape[0]
    positions = dlu.get_pixel_positions((npix, npix))
    grads_vec = np.gradient(array)

    xnorm = positions[1]*grads_vec[0]
    ynorm = positions[0]*grads_vec[1]
    return np.square(xnorm + ynorm)


def RWGE(array, epsilon = 1e-8):
    """Calcuates the spatial radially weighted gradient energy of the array."""
    npix = array.shape[0]
    positions = dlu.get_pixel_positions((npix, npix))
    radii = dlu.get_pixel_positions((npix, npix), polar=True)[0]
    radii_norm = positions/(radii + epsilon)
    grads_vec = np.gradient(array)

    xnorm = radii_norm[1]*grads_vec[0]
    ynorm = radii_norm[0]*grads_vec[1]
    return np.square(xnorm + ynorm)


def radial_mask(npixels, rmin, rmax):
    """Binary radial mask, masking out radii below rmin, and above rmax."""
    radii = dlu.get_pixel_positions((npixels, npixels), polar=True)[0]
    return np.asarray((radii < rmax) & (radii > rmin), dtype=float)


def pix_per_fringe(model):
    optics = model.optics
    source = model.source

    fringe_size_rad = source.wavelengths.max() / optics.diameter
    fringe_size_arcsec = dlu.radians_to_arcseconds(fringe_size_rad)
    true_pixel_scale = optics.psf_pixel_scale / optics.psf_oversample
    return fringe_size_arcsec / true_pixel_scale


def fringe_per_pix(model):
    optics = model.optics
    source = model.source

    fringe_size_rad = source.wavelengths.max() / optics.diameter
    fringe_size_arcsec = dlu.radians_to_arcseconds(fringe_size_rad)
    true_pixel_scale = optics.psf_pixel_scale / optics.psf_oversample
    return true_pixel_scale / fringe_size_arcsec