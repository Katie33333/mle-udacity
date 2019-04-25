## Regression - goal is to predict values
## Classification - yes or no (spam or not, will a user dl an app or not, sick or not)
## Linear Regression Quiz
```
# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
```

**Higher Dimensions** - Example was predicting housing prices but instead of just basing it off size, school quality was introduced, which made the line a plane

**Multiple Linear Regression** - Use multiple linear regression to predict life expectancy using BMI and heart rate

**Predictor/Independent variable** - In the example, we used BMI (the predictor) to predict the life expectancy, which is the **dependent variable**

## Multiple Linear Regression Quiz
```
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load the data from the boston house-prices dataset 
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
# TODO: Fit the model and Assign it to the model variable
model = LinearRegression()
model.fit(x, y)

# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
prediction = model.predict(sample_house)
```

Why can't we use the math to solve the unknowns instead of going through many gradient descent steps?  Because with more than 2 dimensions in the input, n, then we'd have n equations with n unknowns. Solving a system of n equations and n unknowns is very expensive.  If n is big then at some point in our solution we have to invert an n by n matrix.  Inverting a huge matrix is something that take a lot of time and computing power which is not feasible so we use gradient descent.  It will get close to the exact answer which should fit our solution prettty well.  If we had infinate computer power, we'd solve it in one step with linear algebra

* Linear regression works best when the data is linear - Linear regression produces a straight line model from the training data. If the relationship in the training data is not really linear, you'll need to either make adjustments (transform your training data), add features (we'll come to this next), or use another kind of model.  
* Linear regression is sensitive to outliers - In most circumstances, you'll want a model that fits most of the data most of the time, so watch out for outliers!

**Polynomial Regression** - when data points look like a curvy line

**Regularization** - works for regression and classification models - Improves models and makes sure they don't overfit.  Take the complexity of the model into account when calculating the error.  Remember the example of a line (that misclassified a couple points) and the polynomial which didn't misclassify any points but was more complex.  We want to choose the line one that will have a lower combined error. Simple models tend to generalize better so that's what we want.

**L1 Regularization** - Add absolute value of the coefficients to the error

**L2 Regularization** - Add the sqares of the coeffiecients to the error

**Simple versus Complex** - Model to send rocket to the moon or a medical model - little room for error so it's okay if it's more complex versus a video recommendation system that's ok if there's error but needs to run fast against big data.  It requires simplicity.  The **lambda** parameter lets us tune how much complexity is acceptable.  If we have a large lambda, we're punishing the complex model by a lot thus picking a simpler model.  A small lambda punishes complexity by a small amount meaning we're okay with having more complex models.
* See L1 versus L2 regularization screen shot

## Classification Algorithms

**Perceptron Algorithm** - basis for neural networks

AND and OR perceptrons have different weights and biases

XOR perceptron returns a true if either of the inputs are true

**Perceptron trick** - misclassified points want the line to come closer

Example:
```
import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b

    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
```
## Decision Trees
**Entropy** - how many different ways (combinations) can organize something?  A lot = high entropy.  (Ice, water, vapor example).  The more homogenious the set is, the less ways there are to organize it, and the less entropy it has

* For every node in the decision tree we can calculate the entropy of the parent node and then we calculate the entropy of the two children.  **Information gain** is the `etropy(parent) - 0.5 [Entropy(child1) + Entropy(child2)]`

* Decision trees tend to overfit

* Random Forest - pick a subset of features and build a tree.  Pick different features, and build another tree and so on.  When we get a new data point, let all trees make a prediction and pick the one that comes up the most

* Large depth very often causes overfitting, since a tree that is too deep, can memorize the data. Small depth can result in a very simple model, which may cause underfitting.
* Small minimum samples per split may result in a complicated, highly branched tree, which can mean the model has memorized the data, or in other words, overfit. Large minimum samples may result in the tree not having enough flexibility to get built, and may result in underfitting.
