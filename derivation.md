# Gradient derivation

This page details the derivation of the gradient for the loss function including the discrimination penalty.

## Logisitic regression gradient

The standard loss function is as follows;

<img src="https://render.githubusercontent.com/render/math?math=%24J(%5Ctheta)%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5E%7Bm%7D_%7Bi%3D1%7Dy%5E%7Bi%7D%5Clog%7B(h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%7D%20%2B%20(1%20-%20y%5E%7Bi%7D)%5Clog%7B(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%7D%24">

$J(\theta) = \frac{1}{m}\sum^{m}_{i=1}y^{i}\log{(h_{\theta}(x^{i}))} + (1 - y^{i})\log{(1 - h_{\theta}(x^{i}))}$

with the following gradient;

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20J(%5Ctheta)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum%5E%7Bm%7D_%7Bi%3D1%7D(h_%7B%5Ctheta%7D(x%5E%7Bi%7D))%20-%20y%5E%7Bi%7D)%24">

$\frac{\delta J(\theta)}{\delta\theta_{j}} = \frac{1}{m}\sum^{m}_{i=1}(h_{\theta}(x^{i})) - y^{i})$

Also of note is the partial derivative of model predictions w.r.t <img src="https://render.githubusercontent.com/render/math?math=%5Ctheta%5E%7Bj%7D">;

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)%7D%7B%5Cdelta%5Ctheta_%7Bj%7D%7D%20%3D%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D)(1%20-%20h_%7B%5Ctheta%7D(x%5E%7Bi%7D))x_%7Bj%7D%5E%7Bi%7D%24">

$\frac{\delta h_{\theta}(x^{i})}{\delta\theta_{j}} = h_{\theta}(x^{i})(1 - h_{\theta}(x^{i}))x_{j}^{i}$

## Discrimination aware loss function

`Discrimination aware loss function = Logistic regression cost function + Discrimination Penalty`

Let 
- `P` be the `Discrimination Penalty` 
- `G1` and `G2` be the levels of a categorical variable `G`, that we want to prevent discrimination by
- `d` be the difference in the average prediction between groups `G1` and `G2`

then

<img src="https://render.githubusercontent.com/render/math?math=%24P%20%3D%20%5Clambda%5Clog%7B(1%20-%20d%5E%7B2%7D)%7D%24"> where <br>
<img src="https://render.githubusercontent.com/render/math?math=%24d%20%3D%20%5Cfrac%7B1%7D%7BN_%7BG1%7D%7D%5Csum_%7Bi%20%5Cin%20G1%7Dh_%7B%5Ctheta%7D(x%5E%7Bi%7D)%20-%20%5Cfrac%7B1%7D%7BN_%7BG2%7D%7D%5Csum_%7Bi%20%5Cin%20G2%7Dh_%7B%5Ctheta%7D(x%5E%7Bi%7D)%24"> 

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

----

Thanks to [jsfiddle.net](https://jsfiddle.net/8ndx694g/) for converting latex formulae to Github render html. 
