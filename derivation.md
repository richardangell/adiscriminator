# Gradient derivation

`New Cost Function = Logistic Regression Cost Function + Discrimination Penalty` ()

Let <img src="https://render.githubusercontent.com/render/math?math=%24p%24"> be the `Discrimination Penalty`, <img src="https://render.githubusercontent.com/render/math?math=%24G1%24"> and <img src="https://render.githubusercontent.com/render/math?math=%24G2%24"> be the levels of a categorical variable, that we want to prevent discrimination by.

Let <img src="https://render.githubusercontent.com/render/math?math=%24AVE_%7Bgroup%7D%24"> be the average prediction for observations in `group`.

<img src="https://render.githubusercontent.com/render/math?math=%24p%20%3D%20-log(1%20-%20(AVE_%7BG1%7D%20-%20AVE_%7BG2%7D)%5E%7B2%7D)%20%3D%20-log(1%20-%20a%5E2)%24">

where <img src="https://render.githubusercontent.com/render/math?math=%24a%20%3D%20AVE_%7BG1%7D%20-%20AVE_%7BG2%7D%24"> is the difference between average predictions for <img src="https://render.githubusercontent.com/render/math?math=%24G1%24"> and <img src="https://render.githubusercontent.com/render/math?math=%24G2%24">.

The gradient of the penalty term w.r.t. each coefficient is <img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20p%7D%7B%5Cdelta%20%5Cbeta_%7Bi%7D%7D%24">;

<img src="https://render.githubusercontent.com/render/math?math=%24%20%5Cfrac%7B%5Cdelta%20p%7D%7B%5Cdelta%20%5Cbeta_%7Bi%7D%7D%20%20%3D%20%5Cfrac%7B%5Cdelta%20p%7D%7B%5Cdelta%20%20a%7D%20%5Cfrac%7B%5Cdelta%20a%7D%7B%5Cdelta%20%5Cbeta_%7Bi%7D%7D%24">

<img src="https://render.githubusercontent.com/render/math?math=%24%20%3D%20%5Cfrac%7B2a%7D%7B1%20-%20a%5E2%7D%5Cfrac%7B%5Cdelta%20a%7D%7B%5Cdelta%20%5Cbeta_%7Bi%7D%7D%24">

Now let <img src="https://render.githubusercontent.com/render/math?math=%24I_%7Bj%2C%20group%7D%20%3D%201%24"> if <img src="https://render.githubusercontent.com/render/math?math=%24j%5Cin%20group%24"> and 0 otherwise, be a row-wise indicator for group membership i.e. <img src="https://render.githubusercontent.com/render/math?math=%24I_%7Bj%2C%20G2%7D%20%3D%201%24"> means observation <img src="https://render.githubusercontent.com/render/math?math=%24j%24"> is in <img src="https://render.githubusercontent.com/render/math?math=%24G2%24"> (or takes the value <img src="https://render.githubusercontent.com/render/math?math=%24G2%24"> for the categorical variable we are trying to prevent discrimination over). 

<img src="https://render.githubusercontent.com/render/math?math=%24a%20%3D%20(%5Cfrac%7B%5Csum_%7Bj%7DI_%7Bj%2C%20G1%7D.sigmoid(%5Csum%5Cbeta_%7Bi%7Dx_%7Bi%2Cj%7D)%7D%7B%5Csum%7BI_%7Bj%2C%20G1%7D%7D%7D)%20-%20(%5Cfrac%7B%5Csum_%7Bj%7DI_%7Bj%2C%20G2%7D.sigmoid(%5Csum%5Cbeta_%7Bi%7Dx_%7Bi%2Cj%7D)%7D%7B%5Csum%7BI_%7Bj%2C%20G2%7D%7D%7D)%24">

Note <img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%5Csigma(x)%7D%7B%5Cdelta%20x%7D%20%3D%20%5Csigma(x).(1-%5Csigma(x))%24%20where%20%24%5Csigma(x)%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-x%7D%7D%24"> from [here](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x). 

<img src="https://render.githubusercontent.com/render/math?math=%24%5Cfrac%7B%5Cdelta%20a%7D%7B%5Cdelta%20%5Cbeta_%7Bi%7D%7D%20%3D%20%5Csum_%7Bj%7DI_%7Bj%2CG1%7D.x_%7Bi%2Cj%7D.sigmoid(%5Csum%5Cbeta_%7Bi%7Dx_%7Bi%2Cj%7D).(1%20-%20sigmoid(%5Csum%5Cbeta_%7Bi%7Dx_%7Bi%2Cj%7D))%2F%5Csum_%7Bj%7DI_%7Bj%2CG1%7D%20%2B%20%5Csum_%7Bj%7DI_%7Bj%2CG2%7D.x_%7Bi%2Cj%7D.sigmoid(%5Csum%5Cbeta_%7Bi%7Dx_%7Bi%2Cj%7D).(1%20-%20sigmoid(%5Csum%5Cbeta_%7Bi%7Dx_%7Bi%2Cj%7D))%2F%5Csum_%7Bj%7DI_%7Bj%2CG2%7D%20%24">


