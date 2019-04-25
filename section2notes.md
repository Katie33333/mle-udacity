
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
