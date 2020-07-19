# Gradient derivation

This page details the derivation of the gradient for the loss function including the group difference penalty.

## Logisitic regression cost function and gradient

The standard loss function is as follows;

<img src="https://render.githubusercontent.com/render/math?math=%24J(%5Ctheta)%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5E%7Bm%7D_%7Bi%3D1%7Dy%5E%7Bi%7D%5Clog%7B(h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%7D%20%2B%20(1%20-%20y%5E%7Bi%7D)%5Clog%7B(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%7D%24">

$J(\theta) = \frac{1}{m}\sum^{m}_{i=1}y^{i}\log{(h_{\theta}(x^{i}))} + (1 - y^{i})\log{(1 - h_{\theta}(x^{i}))}$

with the following gradient;

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20J(%5Ctheta)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5E%7Bm%7D_%7Bi%3D1%7D(h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20-%20y%5E%7Bi%7D)%24">

$\frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m}\sum^{m}_{i=1}(h_{\theta}(x^{i})) - y^{i})$

Also of note is the partial derivative of model predictions w.r.t <img src="https://render.githubusercontent.com/render/math?math=%24%5Ctheta_%7Bj%7D%24">;

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%24">

$\frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} = h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i}$

## Group difference aware loss function

`Group difference aware loss function = Logistic regression cost function + Group Difference Penalty`

Let 
- `P` be the `Group Difference Penalty` 
- `G1` and `G2` be the levels of a categorical variable `G`, that we want to prevent difference by (on average)
- `d` be the difference in the average prediction between groups `G1` and `G2` in `G`

then

<img src="https://render.githubusercontent.com/render/math?math=%24P%20%3D%20%5Clambda%5Clog%7B(1%20-%20d%5E%7B2%7D)%7D%24"> where <br>
<img src="https://render.githubusercontent.com/render/math?math=%24d%20%3D%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7Dh_%7B%5Ctheta%7D(x%5E%7Bi%7D)%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7Dh_%7B%5Ctheta%7D(x%5E%7Bi%7D)%24"> 

## Derivation of penalty gradient

Now 

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20P%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B%5Cdelta%20P%7D%7B%5Cdelta%20d%7D%20%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%24">

$\frac{\delta P}{\delta\theta_{j}} = \frac{\delta P}{\delta d} \frac{\delta d}{\delta\theta_{j}}$

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20P%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B-2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%24">

$\frac{\delta P}{\delta\theta_{j}} = \frac{-2\lambda d}{1 - d^{2}} \frac{\delta d}{\delta\theta_{j}}$

and

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%24">

$\frac{\delta d}{\delta\theta_{j}} = \frac{1}{N_{G1}}\sum_{i \in G1} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} - \frac{1}{N_{G2}}\sum_{i \in G2} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}}$

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%24">

$\frac{\delta d}{\delta\theta_{j}} = \frac{1}{N_{G1}}\sum_{i \in G1} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} - \frac{1}{N_{G2}}\sum_{i \in G2} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i}$

So the gradient of the penalty term w.r.t. each coefficient is

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20P%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B-2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cleft%5B%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20%5Cright%5D%24">

$\frac{\delta P}{\delta\theta_{j}} = \frac{-2\lambda d}{1 - d^{2}} \left[ \frac{1}{N_{G1}}\sum_{i \in G1} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} - \frac{1}{N_{G2}}\sum_{i \in G2} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} \right]$

## Derivation of second derivative of penalty

The the second derivative is

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%5E%7B2%7D%20P%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%5E%7B2%7D%7D%20%3D%20%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B-2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cright).%5Cleft%5B%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cright%5D%20-%20%5Cfrac%7B2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D.%20%5Cleft%5B%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20%5Cfrac%7B%5Cdelta%5E%7B2%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%5E%7B2%7D%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20%5Cfrac%7B%5Cdelta%5E%7B2%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%5E%7B2%7D%7D%20%5Cright%5D%20%24">

$\frac{\delta^{2} P}{\delta\theta_{j}^{2}} = \frac{\delta}{\delta\theta_{j}} \left( \frac{-2\lambda d}{1 - d^{2}} \right).\left[ \frac{1}{N_{G1}}\sum_{i \in G1} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} - \frac{1}{N_{G2}}\sum_{i \in G2} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} \right] - \frac{2\lambda d}{1 - d^{2}}. \left[ \frac{1}{N_{G1}}\sum_{i \in G1} \frac{\delta^{2} h_{\theta}(x^{i})}{\delta\theta_{j}^{2}} - \frac{1}{N_{G2}}\sum_{i \in G2} \frac{\delta^{2} h_{\theta}(x^{i})}{\delta\theta_{j}^{2}} \right] $

or 

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%5E%7B2%7D%20P%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%5E%7B2%7D%7D%20%3D%20%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B-2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cright).%5Cleft%5B%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20%5Cright%5D%20-%20%5Cfrac%7B2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D.%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20%5Cright)%20%24">

$\frac{\delta^{2} P}{\delta\theta_{j}^{2}} = \frac{\delta}{\delta\theta_{j}} \left( \frac{-2\lambda d}{1 - d^{2}} \right).\left[ \frac{1}{N_{G1}}\sum_{i \in G1} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} - \frac{1}{N_{G2}}\sum_{i \in G2} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} \right] - \frac{2\lambda d}{1 - d^{2}}.\frac{\delta}{\delta\theta_{j}} \left( \frac{1}{N_{G1}}\sum_{i \in G1} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} - \frac{1}{N_{G2}}\sum_{i \in G2} h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} \right) $

So starting with the left side term

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B-2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cright)%20%3D%20-2%5Clambda%20%5Cleft(%20%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D.%5Cfrac%7B1%7D%7B1%20-%20d%5E%7B2%7D%7D%20%2B%20d.%20%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B1%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cright)%20%5Cright)%24">

$\frac{\delta}{\delta\theta_{j}} \left( \frac{-2\lambda d}{1 - d^{2}} \right) = -2\lambda \left( \frac{\delta d}{\delta\theta_{j}}.\frac{1}{1 - d^{2}} + d. \frac{\delta}{\delta\theta_{j}} \left( \frac{1}{1 - d^{2}} \right) \right)$

And 

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B1%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cright)%20%3D%20%5Cfrac%7B-1%7D%7B(1%20-%20d%5E%7B2%7D)%5E%7B2%7D%7D.%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%201%20-%20d%5E%7B2%7D%20%5Cright)%20%3D%20%5Cfrac%7B-1%7D%7B(1%20-%20d%5E%7B2%7D)%5E%7B2%7D%7D.-2d.%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B2d%7D%7B(1%20-%20d%5E%7B2%7D)%5E%7B2%7D%7D.%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%24">

$\frac{\delta}{\delta\theta_{j}} \left( \frac{1}{1 - d^{2}} \right) = \frac{-1}{(1 - d^{2})^{2}}.\frac{\delta}{\delta\theta_{j}} \left( 1 - d^{2} \right) = \frac{-1}{(1 - d^{2})^{2}}.-2d.\frac{\delta d}{\delta\theta_{j}} = \frac{2d}{(1 - d^{2})^{2}}.\frac{\delta d}{\delta\theta_{j}}$

Hence 

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B-2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cright)%20%3D%20-2%5Clambda%20%5Cleft(%20%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D.%5Cfrac%7B1%7D%7B1%20-%20d%5E%7B2%7D%7D%20%2B%20d.%5Cfrac%7B2d%7D%7B(1%20-%20d%5E%7B2%7D)%5E%7B2%7D%7D.%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cright)%20%3D%20-2%5Clambda%20%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20%5Cfrac%7B1%7D%7B1%20-%20d%5E%7B2%7D%7D%20%2B%20%5Cfrac%7B2d%5E%7B2%7D%7D%7B(1%20-%20d%5E%7B2%7D)%5E%7B2%7D%7D%20%5Cright)%20%3D%20-2%5Clambda%20%5Cfrac%7B1%20%2B%20d%5E%7B2%7D%7D%7B(1%20-%20d%5E%7B2%7D)%5E%7B2%7D%7D%20%5Cfrac%7B%5Cdelta%20d%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%24">

$\frac{\delta}{\delta\theta_{j}} \left( \frac{-2\lambda d}{1 - d^{2}} \right) = -2\lambda \left( \frac{\delta d}{\delta\theta_{j}}.\frac{1}{1 - d^{2}} + d.\frac{2d}{(1 - d^{2})^{2}}.\frac{\delta d}{\delta\theta_{j}} \right) = -2\lambda \frac{\delta d}{\delta\theta_{j}} \left( \frac{1}{1 - d^{2}} + \frac{2d^{2}}{(1 - d^{2})^{2}} \right) = -2\lambda \frac{1 + d^{2}}{(1 - d^{2})^{2}} \frac{\delta d}{\delta\theta_{j}}$

Then going back to the right side term

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%20%5Cright)%20%3D%20x_%7Bj%7D%5E%7Bi%7D%20%5Cfrac%7B%5Cdelta%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cleft(%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20%5Cright)%24">

$\frac{\delta}{\delta\theta_{j}} \left( h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i} \right) = x_{j}^{i} \frac{\delta}{\delta\theta_{j}} \left( h_{\theta}(x^{i})(1 - h_{\theta}(x^{i})) \right)$

<img src="https://render.githubusercontent.com/render/math?math=%24%20%3D%20x_%7Bj%7D%5E%7Bi%7D%20%5Cleft(%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20%2B%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%5Cfrac%7B%5Cdelta%20(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cright)%20%3D%20x_%7Bj%7D%5E%7Bi%7D%20%5Cleft(%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cright)%20%3D%20x_%7Bj%7D%5E%7Bi%7D%20(1%20-%202h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%24">

$ = x_{j}^{i} \left( \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}}(1 - h_{\theta}(x^{i})) + h_{\theta}(x^{i})\frac{\delta (1 - h_{\theta}(x^{i}))}{\delta\theta_{j}} \right) = x_{j}^{i} \left( \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}}(1 - h_{\theta}(x^{i})) - h_{\theta}(x^{i})\frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} \right) = x_{j}^{i} (1 - 2h_{\theta}(x^{i})) \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}}$

Putting it all together

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%5E%7B2%7D%20P%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%5E%7B2%7D%7D%20%3D%20-2%5Clambda%20%5Cfrac%7B1%20%2B%20d%5E%7B2%7D%7D%7B(1%20-%20d%5E%7B2%7D)%5E%7B2%7D%7D.%5Cleft%5B%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%5Cright%5D%5E%7B2%7D%20-%20%5Cfrac%7B2%5Clambda%20d%7D%7B1%20-%20d%5E%7B2%7D%7D%20%5Cleft%5B%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20(1%20-%202h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20x_%7Bj%7D%5E%7Bi%7D%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7D%20%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20(1%20-%202h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20x_%7Bj%7D%5E%7Bi%7D%20%5Cright%5D%20%24">

$\frac{\delta^{2} P}{\delta\theta_{j}^{2}} = -2\lambda \frac{1 + d^{2}}{(1 - d^{2})^{2}}.\left[ \frac{1}{N_{G1}}\sum_{i \in G1} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} - \frac{1}{N_{G2}}\sum_{i \in G2} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} \right]^{2} - \frac{2\lambda d}{1 - d^{2}} \left[ \frac{1}{N_{G1}}\sum_{i \in G1} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} (1 - 2h_{\theta}(x^{i})) x_{j}^{i} - \frac{1}{N_{G2}}\sum_{i \in G2} \frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} (1 - 2h_{\theta}(x^{i})) x_{j}^{i} \right] $

----

Thanks to [jsfiddle.net](https://jsfiddle.net/8ndx694g/) for converting latex formulae to Github render html. 
