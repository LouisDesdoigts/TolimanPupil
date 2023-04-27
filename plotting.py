from matplotlib import pyplot as plt
from matplotlib import colormaps
from bayes import get_covaraince
from chainconsumer import ChainConsumer

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120
cmap = colormaps['inferno']
cmap.set_bad('k',0.8)

def plot_mask_point(model):
    source = model.source
    binary_phase = model.mask.get_binary_phase()
    inv_support = np.where(model.aperture <= 0.5)
    mask = binary_phase.at[inv_support].set(np.nan)
    psf = model.propagate(source.wavelengths)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap=cmap)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(psf)
    plt.colorbar()
    plt.show()


def plot_covariances(models, parameters, log_flux=6):
    """Plots the covarinaces of the given models."""
    if not isinstance(models, list):
        models = [models]
    
    covs = [get_covaraince(model, parameters, log_flux=log_flux) \
        for model in models]

    if 'aberrations.coefficients' in parameters:
        param_names = [map_name(p, latex=True) for p in parameters[:-1]]
        param_names += map_name(parameters[-1], latex=True, model=models[0])
    X = np.zeros(len(param_names))

    c = ChainConsumer()
    for cov in covs:
        c.add_covariance(X, cov, parameters=param_names)
    c.configure(serif=True, shade=True, bar_shade=True, shade_alpha=0.2, 
        spacing=1., max_ticks=3)

    fig = c.plotter.plot()


def map_name(name, latex=False, model=None):
    """Maps parameter names to shortened strings for plotting."""
    # Separation
    if name == 'separation':
        if latex:
            return r'$r$'
        else:
            return 'r'

    # X Position
    if name == 'x_position':
        if latex:
            return r'$x$'
        else:
            return 'x'

    # Y Position
    if name == 'y_position':
        if latex:
            return r'$y$'
        else:
            return 'y'
    
    # Position angle
    if name == 'position_angle':
        if latex:
            return r'$\phi$'
        else:
            return 'phi'
    
    # Log Flux
    if name == 'log_flux':
        if latex:
            return r'$\log_{10} F$'
        else:
            return 'log_flux'
    
    # Contrast
    if name == 'contrast':
        if latex:
            return r'$C$'
        else:
            return 'contrast'
    
    # Wavelengths
    if name == 'wavelengths':
        if latex:
            return r'$\lambda$'
        else:
            return 'lambda'
    
    # Pixel Scale
    if name == 'psf_pixel_scale':
        if latex:
            return r'$pix_scale$'
        else:
            return 'pix_scale'
    
    # Zernike Coefficients
    if name == 'aberrations.coefficients':
        if model is None:
            raise ValueError("Must provide a model to map Zernike coefficients")
        nterms = len(model.aberrations.coefficients)
        if latex:
            return [r'$Z_{}$'.format(i) for i in range(4, 4 + nterms)]
        else:
            return ['Z{}'.format(i) for i in range(4, 4 + nterms)]
