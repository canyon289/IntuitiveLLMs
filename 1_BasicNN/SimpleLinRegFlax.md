---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
# Intuitive AI: Session 2
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Study Club Overview
* Today: Get into the details of NN, Flax, and model thinking
* Next Session: NN Architecture overview. CNN, RNN, Transformers
* Implementing (but more like copying) a Basic GPT
* TBD (and new!): LLM Ecosystem
* TBD: Shaping LLMs to do what we want
* Maybe more stuff later
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Study Club Overview Annotated
* Today: Get into the details of NN, Flax, and model thinking
  * What are these things really at their core?
* Next Session: NN Architecture overview. CNN, RNN, Transformers
  * How can we piece them together?
* Basic Languge Models and implementing (but more like copying) a Basic GPT
  * What does a modern architecture look like for something we can train ourselves
* TBD: LLM Ecosystem
  * A bunch of hot takes from Ravin on the landscape of closed source vs open source vs sorta open LLMs
* Shaping LLMs to do what we want
  * Making these things work for us
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Today
Understanding every part of a basic NN
<!-- #endregion -->

  * The basic math (it's way easier than you may think)
  * The code (the more important one so we don't get overwhelemd later)
  * The mathematical ideas (I don't think this gets covered enough)

<!-- #region slideshow={"slide_type": "slide"} -->
## Here's our our plan
1. Create some synthetic X and Y data generated from a linear model
  * Foundational scipy and numpy usage
2. Use Neural Nets to recover the parameters to show how they work
  * Build this in flax
3. Do the same with a non linear data generating function
  * Show that NNs can fit "anything"
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## What to remember
* General intuition of how NN works
* Flax model definition syntax
* How parameters are estimated
  * Loss Function
  * Usage of optax
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## References
* NN are just lin reg: https://joshuagoings.com/2020/05/05/neural-network/
* Why we want multi layer NN https://lightning.ai/pages/courses/deep-learning-fundamentals/training-multilayer-neural-networks-overview/4-5-multilayer-neural-networks-for-regression-parts-1-2/
* Flax Docs https://flax.readthedocs.io/en/latest/
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Generating some data
So we have something to do
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Our data generating function
<!-- #endregion -->

$$ y = mx + b $$


$$ y = m_0 * x_0 + m_1 * x_1 + b $$

<!-- #region slideshow={"slide_type": "slide"} -->
## Set Coefficients and constants
This is what we're going to recover
<!-- #endregion -->

```python
m_0, m_1 = 1.1, 2.1
intercept = bias = 3.1
```

```python
# The shape needs to be (1,2). You'll see why below
coefficients = np.array([[m_0, m_1]])
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Generate some observed x data
We need two dimensions because we have two coefficients
<!-- #endregion -->

```python
x_0, x_1 = 0, 1
coefficients[0][0] * x_0 + coefficients[0][1] * x_1 + bias
```

```python
x_0, x_1 = 1, 0
coefficients[0][0] * x_0 + coefficients[0][1] * x_1 + bias
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Plot our function grid
<!-- #endregion -->

```python
from mpl_toolkits.mplot3d import Axes3D
# Define grid range
x_0_range = np.linspace(-5, 5, 20)
x_1_range = np.linspace(-5, 5, 20)
X_0, x_1 = np.meshgrid(x_0_range, x_1_range)

# Calculate corresponding z coordinates

Y = coefficients[0][0] * X_0 + coefficients[0][1] * x_1 + bias

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot plane
ax.plot_surface(X_0, x_1, Y)

# Set axis labels
ax.set_xlabel('X_0')
ax.set_ylabel('X_1')
ax.set_zlabel('Y')

ax.mouse_init()
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Generate Random X_0, X_1 points
This is what we'd get in real life. Not everything is on a grid
<!-- #endregion -->

```python
rng = np.random.default_rng(12345)
x_obs = rng.uniform(-10, 10, size=(200,2))
x_obs[:10]
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Side Track: Do some matrix multplication
<!-- #endregion -->

```python
y_obs = coefficients[0][0] * x_obs[0][0] + coefficients[0][1] * x_obs[0][1] + bias
y_obs
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Practical Tip 1: You're going to check shapes, so, much
With neural nets these matter a lot
<!-- #endregion -->

```python
coefficients.shape, x_obs.shape
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Matrix Multiplication
(1,2) @ (2, 200) = (1,200) 
<!-- #endregion -->

```python
# Typically done this way in NN literature so we get a column vector 
y_obs = (x_obs @ coefficients.T) + bias
y_obs[:5]
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Einsum is really nice
<!-- #endregion -->

```python
y_obs = np.einsum('ij,kj->ki', coefficients, x_obs) + bias
y_obs[:5]
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Lets make add some noise to Y 
In real life we'll never perfectly get our exact measurements. We should add to add some noise
<!-- #endregion -->

```python
noise_sd = .5
y_obs_noisy = y_obs + stats.norm(0, noise_sd).rvs(y_obs.shape)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## What we did until here
* We decided our data generated function is a two coefficient regression with a bias
  * $$ y = m_0 * x_0 + m_1 * x_1 + b $$
* Picked some arbitrary parameters
  * `m_0, m_1 = 1.1, 2.1`
  * `intercept = bias = 3.1`
* We picked some random x_0 and x_1 observations and calculated our y1_observations
  * In the real world we'd have observed this
  * weight = m1*height + m2+width
* We generated some random x_0 and x_1 points
  * Used that to calculate Y
* Added some random noise to Y
  * The real world is messy
**No modeling has been completed yet**
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Reminder of our Goal
The goal here is to figure out the **coefficients m_0, m_1, bias** using our **observed x_0, x_! data**
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## How can we we do this?
Lots of ways
1. "Linear Regression"
2. "Machine Learning 
3. Bayesian Regression
4. Neural Nets
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Bayesian Model
<!-- #endregion -->

```python
import bambi as bmb
import arviz as az
```

```python
df = pd.DataFrame({"y":y_obs_noisy[:, 0], "x0":x_obs[:,0], "x1":x_obs[:,1]})
model = bmb.Model('y ~ x0 + x1', data=df)
idata = model.fit()
```

```python
model = bmb.Model('y ~ x0 + x1', data=df)
idata = model.fit()
```

```python
az.plot_trace(idata);
```

```python
az.summary(idata)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Exercise for you to try
Fit this this linear regression using
1. A traditional solver like statmodels
2. Any ML method from scikit learn
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## The point I'm trying to make
These single terms often conflate three things
1. The type of data
2. The model architecture
3. The parameter estimation method
4. The libraries used
5. The purpose of the analysis

When learning its important to understand the distinction of these parts. In this study club we'll be weaving through these topics
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Neural Nets are just lin reg at their core as well
<!-- #endregion -->

<!-- #region -->


<center>
  <img src="img/LinearReg.png" style="height:400"; />
</center>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Neural Nets (in Flax)
<!-- #endregion -->

```python
import flax.linen as nn
```

```python
class LinearRegression(nn.Module):
    # Define the neural network architecture
    def setup(self):
        """Single output, I dont need to specify shape"""
        self.dense = nn.Dense(features=1)

    # Define the forward pass
    def __call__(self, x):
        y_pred = self.dense(x)
        return y_pred
```

```python
model = LinearRegression()
```

```python
key = jax.random.PRNGKey(0)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Initialize Random Parameters
<!-- #endregion -->

```python
params = model.init(key, x_obs)
params
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Print Model Architecture
<!-- #endregion -->

```python
print(model.tabulate(key, x_obs,
      console_kwargs={'force_terminal': False, 'force_jupyter': True}))
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Estimate y from our random parameters
These numbesr are meaingless. I just want to show you that you can 
* Take any parameter dictionary
* Any X value
* Forward it through the model
<!-- #endregion -->

```python
x1, x2 = 1, 0
model.apply(params, [1,0])
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Estimating our parameters using gradient descent
Basically
1. Start with some parameters that are bad
2. Figure out the direction that is less bad
3. Move that direction

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
### You need a loss function (to define a loss surface)
<!-- #endregion -->

We're going to use L2 loss, also know as mean squared error


https://towardsdatascience.com/a-visual-explanation-of-gradient-descent-methods-momentum-adagrad-rmsprop-adam-f898b102325c

<!-- #region slideshow={"slide_type": "slide"} -->
## L2 Loss by Hand
1. Calculating the y estimate
2. Calculating hte lass
<!-- #endregion -->

```python
m = params["params"]["dense"]["kernel"]
m
```

```python
bias = params["params"]["dense"]["bias"]
```

```python
m = params["params"]["dense"]["kernel"]
y_0_pred = m[0] * x_obs[0][0] + m[1] * x_obs[0][1] + bias
y_0_pred
```

```python
(y_0_pred - y_obs[0])**2 / 2
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Use Optax instead
<!-- #endregion -->

```python
import optax
```

```python
y_pred = model.apply(params, x_obs[0])
y_pred
```

```python
y_pred = model.apply(params, x_obs[0])

optax.l2_loss(y_pred[0], y_obs[0][0])
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Training
Which basically means doing that calculation a bunch of times 
<!-- #endregion -->

```python
from flax.training import train_state  # Useful dataclass to keep train state
```

```python
# Note so jax needs to differentiate this 
@jax.jit
def flax_l2_loss(params, x, y_true): 
    y_pred = model.apply(params, x)
    total_loss = optax.l2_loss(y_pred, y_true).sum()
    
    return total_loss

flax_l2_loss(params, x_obs[0], y_obs[0])
```

<!-- #region slideshow={"slide_type": "slide"} -->
## How far do we step each time?

<!-- #endregion -->

```python
optimizer = optax.adam(learning_rate=0.001)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Storing training state
<!-- #endregion -->

```python
state = train_state.TrainState.create(apply_fn=model, params=params, tx=optimizer)
type(state)
```

```python
state
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Actual training
<!-- #endregion -->

```python
epochs = 10000
_loss = []

for epoch in range(epochs):
    # Calculate the gradient
    loss, grads = jax.value_and_grad(flax_l2_loss)(state.params, x_obs, y_obs_noisy)
    _loss.append(loss)
    # Update the model parameters
    state = state.apply_gradients(grads=grads)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Training Loss
<!-- #endregion -->

```python
fig, ax = plt.subplots()
ax.plot(np.arange(epochs), _loss)
ax.set_xlabel("Step or Epoch")
ax.set_ylabel("Trainng loss");
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Final Parmameters
<!-- #endregion -->

```python
state.params
```

<!-- #region slideshow={"slide_type": "slide"} -->
## What if data is non linear?
Show model can't fit
but then we change architecture and it does
<!-- #endregion -->

```python
x_non_linear = np.linspace(-10, 10, 100)
m = 2
y_non_linear = m*x_non_linear**2
```

```python
fig, ax = plt.subplots()
ax.plot(x_non_linear,y_non_linear);
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Initialize Params for our  model
Shape Problems
<!-- #endregion -->

```python
params = model.init(key, x_non_linear)
params["params"]["dense"]["kernel"].shape
```

<!-- #region slideshow={"slide_type": "slide"} -->
## We need to reshape X
<!-- #endregion -->

```python
x_non_linear[..., None][:5]
```

```python
params = model.init(key, x_non_linear[..., None])
params
```

```python
state = train_state.TrainState.create(apply_fn=model, params=params, tx=optimizer)
type(state)
```

```python
epochs = 10000
_loss = []

for epoch in range(epochs):
    # Calculate the gradient, shapes are really annoying
    loss, grads = jax.value_and_grad(flax_l2_loss)(state.params, x_non_linear[..., None], y_non_linear[..., None])
    _loss.append(loss)
    # Update the model parameters
    state = state.apply_gradients(grads=grads)
```

```python
state.params
```

```python
y_pred = model.apply(state.params, x_non_linear[..., None])
```

```python
y_pred[:5]
```

<!-- #region slideshow={"slide_type": "slide"} -->
## This is a terrible fit
<!-- #endregion -->

```python
fig, ax = plt.subplots()
ax.plot(x_non_linear, y_non_linear, label="Actual");
ax.plot(x_non_linear, y_pred, label="Predicted");
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Nonlinear Regression
<!-- #endregion -->

```python
class NonLinearRegression(nn.Module):
    # Define the neural network architecture
    def setup(self):
        """Single output, I dont need to specify shape"""
        self.hidden_layer_1 = nn.Dense(features=10)
        self.hidden_layer_2 = nn.Dense(features=10)
        self.dense_out = nn.Dense(features=1)

    # Define the forward pass
    def __call__(self, x):
        hidden_x_1 = self.hidden_layer_1(x)
        hidden_x_2 = self.hidden_layer_2(hidden_x_1)
        y_pred = self.dense_out(hidden_x_2)
        return y_pred
```

```python
class NonLinearRegression(nn.Module):
    
    # Define the neural network architecture
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)                 # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)       # shape inference
        return x
```

```python
model = NonLinearRegression()
params = model.init(key, x_non_linear[..., None])
params
```

```python
state = train_state.TrainState.create(apply_fn=model, params=params, tx=optimizer)
```

```python
epochs = 10000
_loss = []

for epoch in range(epochs):
    # Calculate the gradient. Also shapes are really annoying
    loss, grads = jax.value_and_grad(flax_l2_loss)(state.params, x_non_linear[..., None], y_non_linear[..., None])
    _loss.append(loss)
    # Update the model parameters
    state = state.apply_gradients(grads=grads)
```

```python
state;
```

```python
fig, ax = plt.subplots()
ax.plot(np.arange(epochs), _loss)
ax.set_xlabel("Step or Epoch")
ax.set_ylabel("Trainng loss");
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Calculate Predictions
<!-- #endregion -->

```python
y_pred_non_linear = model.apply(state.params, x_non_linear[..., None])
```

## Plot Predictions

```python
fig, ax = plt.subplots()
ax.plot(x_non_linear, y_non_linear, label="Actual");
ax.plot(x_non_linear, y_pred, label="Predicted");
ax.plot(x_non_linear, y_pred_non_linear, label="Predicted", ls='--', lw=4)
plt.legend();
```
