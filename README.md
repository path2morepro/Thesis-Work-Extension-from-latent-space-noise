# Noise Latent space

This repo contains the files for the research project on forecasting in noise latent spaces in course 732A76.

Example code for inverting and sample is in `playground.ipynb`.

To generate more training data run `sqg_nature_run.py` and then `generate_inverted_timeseries.py`

To sample you need the following packages
* pytorch
* numpy
* matplotlib

To generate new training data you also need
* pyfftw
* netcdf4

# Where the noise latent space comes from?
 
## Preliminaries
![z0 ~ N(0, I) and zt is the original data](image.png)

## What is the model training?

The model takes as input
$z_t = (1-t) * z_0 + t * z_1$, an interpolated intermediate state between $z_0$ and $z_1$.
The training target is
$\frac{d z_t}{d t}$, which in this linear interpolation is simply $z_1 - z_0$.
Therefore, the model output is the predicted time-dependent velocity field
$\frac{d z_t}{d t}$.

## ODE composed of model output

In `sampler.py`, the `invert()` function maps a physical field back into latent space.
Under the deterministic setting used for inversion, the process can be viewed as an ODE of the form
$dz = b \, dt$.
The latent representation of each physical frame is obtained by iterating
$z_t = z_{t-1} + dz$
over many small steps.

## Theoretical analysis

The model is trained to fit the velocity field along the interpolation path, not the physical time evolution itself.
This interpolation connects the data distribution to Gaussian noise, $N(0, I)$.
Therefore, as the inversion proceeds, the latent representation is pushed toward the Gaussian prior.
In theory, if the model is accurate enough and the numerical integration is sufficiently fine, the inverted latent space should become increasingly close to samples from $N(0, I)$.
