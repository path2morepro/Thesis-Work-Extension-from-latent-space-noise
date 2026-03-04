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
$dz = b * dt$.
The latent representation of each physical frame is obtained by iterating
$z_t = z_{t-1} + dz$
over many small steps.

# Theoretical analysis

The model is trained to fit the velocity field along the interpolation path, not the physical time evolution itself.
This interpolation connects the data distribution to Gaussian noise, $N(0, I)$.
Therefore, as the inversion proceeds, the latent representation is pushed toward the Gaussian prior.
In theory, if the model is accurate enough and the numerical integration is sufficiently fine, the inverted latent space should become increasingly close to samples from $N(0, I)$.

# Why the latent space may still be useful for forecasting

Although the inversion is designed to push samples toward a Gaussian prior, this does not mean that the latent space is "just noise" in a trivial sense.
Different physical samples can still be mapped to different locations within the shared $N(0, I)$ latent space, so the latent coordinates may retain meaningful structure inherited from the original dynamics.

1. If the temporal evolution in latent space is simpler than in physical space;

2. If the transition $z_t \rightarrow z_{t+1}$ is more linear;

3. If the uncertainty is easier to model there, then forecasting in latent space may still be advantageous.


In that sense, matching a Gaussian prior does not remove the potential forecasting value of the latent representation; it only defines the global distribution that the inverted samples are encouraged to follow.

# What time-aligned interpolation actually do?

What this project mainly studies is time-aligned interpolation. More specifically, interpolation is performed between $z_t$ and $z_{t+\delta t}$ in latent space, and the interpolated latent states are then mapped back to the physical field deterministically. These reconstructed fields are finally compared with the true physical frames.
In the following, I will refer to "time-aligned interpolation" simply as "interpolation". In this project, interpolation is used as a practical diagnostic rather than a definitive test of whether predictable dynamics are preserved in latent space. My original motivation was that if linear or stochastic interpolation can reconstruct intermediate frames with high geometric similarity to the true ones, then the latent representation may still retain some meaningful dynamical structure.

However, the trajectory produced by interpolation is imposed by the interpolation rule itself, whereas the true latent trajectory between two observed states is unknown. For that reason, good interpolation results can indicate geometric consistency or local smoothness, but they do not by themselves prove that the true predictive dynamics have been preserved, simplified, or made easier to forecast in latent space.

# If interpolation cannot achieve any, what is the next step?

The earilst experiment is examining that would forcast latent space eariler than forcast physical space. We can train 2 same models on both datasets, and compare their performance or training.

To achieve that, a proper model is important for this comparison. It must be suitable for both 2 datasets. Here ConvLSTM is chosen in this experiment. Comparing to LSTM, ConvLSTM replace the Hadamard product with convolution operation. Therefore it can memorize spactial information. Sounds very fancy and good for the our data. Here are more reasons about why this model:

1. This model aims to modeling spatiotemperal correlation, especially it has proved good performance on radar or fuild data;

2. There is no need to detect and capture the spatiotemperal correlation in advance. Lots of statistical analysis should be done before we apply any probabilistical model. It's too much for a benchmark check;

3. This model accpet images as input while latent space cannot be encoded into any 1D vector, or just PCA dosen't work on that;

4. This model is more light-weighted, I haven't proven that, but I will(The candidates are PhyDNet, Stochastic Latent Residual Video Prediction, ConvLSTM, PredRNN, Earthformer). 

# Experiment


