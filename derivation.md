# Gradient derivation

`New Cost Function = Logistic Regression Cost Function + Discrimination Penalty (<img src="https://render.githubusercontent.com/render/math?math=%24p%24">)`

Let $G1$ and $G2$ be the (only) levels of a categorical variable, that we want to prevent discrimination by.

Let <img src="https://render.githubusercontent.com/render/math?math=%24AVE_%7Bgroup%7D%24"> be the average prediction for observations in `group`.

<img src="https://render.githubusercontent.com/render/math?math=%24p%20%3D%20-log(1%20-%20(AVE_%7BG1%7D%20-%20AVE_%7BG2%7D)%5E%7B2%7D)%20%3D%20-log(1%20-%20a%5E2)%24">

where $a = AVE_{G1} - AVE_{G2}$ is the difference between average predictions for $G1$ and $G2$.

The gradient of the penalty term w.r.t. each coefficient is $\frac{\delta p}{\delta \beta_{i}}$;

$$ \frac{\delta p}{\delta \beta_{i}}  = \frac{\delta p}{\delta  a} \frac{\delta a}{\delta \beta_{i}}$$

$$ = \frac{2a}{1 - a^2}\frac{\delta a}{\delta \beta_{i}}$$

Now let $I_{j, group} = 1$ if $j\in group$ and $0$ otherwise, be a row-wise indicator for group membership 

i.e. $I_{j, G2} = 1$ means observation $j$ is in $G2$ (or takes the value $G2$ for the categorical variable we are trying to prevent discrimination over). 

$$a = (\frac{\sum_{j}I_{j, G1}.sigmoid(\sum\beta_{i}x_{i,j})}{\sum{I_{j, G1}}}) - (\frac{\sum_{j}I_{j, G2}.sigmoid(\sum\beta_{i}x_{i,j})}{\sum{I_{j, G2}}})$$

Note $\frac{\delta\sigma(x)}{\delta x} = \sigma(x).(1-\sigma(x))$ where $\sigma(x) = \frac{1}{1 + e^{-x}}$ from  [here](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x). 

$$\frac{\delta a}{\delta \beta_{i}} = \sum_{j}I_{j,G1}.x_{i,j}.sigmoid(\sum\beta_{i}x_{i,j}).(1 - sigmoid(\sum\beta_{i}x_{i,j}))/\sum_{j}I_{j,G1} + \sum_{j}I_{j,G2}.x_{i,j}.sigmoid(\sum\beta_{i}x_{i,j}).(1 - sigmoid(\sum\beta_{i}x_{i,j}))/\sum_{j}I_{j,G2} $$


