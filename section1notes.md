# mle-udacity
My projects and notes from the Udacity Machine Learning Engineer nano degree

## Udacity Machine Learning Engineer Nanodegree Notes ##
 
Github: https://github.com/udacity/machine-learning
  
## Section1: Main Algorithms used in Machine Learning
 
**Decision Trees** - for application recommendations
 
Naive Bayes - email spam detection example
P**robability** - 100 emails.  20 out of the 25 spam emails had the word cheap.  5 out of the 74 non-spam emails had the word cheap.
 
20 cheap (.8)
5 cheap (.2)
 
Features:
80% probability that if an email has the word cheap, it’s spam
Spelling mistake - 70%
Missing title - 95%
 
Combine these features to get the **Naive Bayes algorithm**
 
**Gradient Descent algorithm** - Mountain example - Mountain is the problem - take tiny steps to get you to the solution which is the bottom of the mountain
 
**Linear Regression** - Draw the best fitting line through the data to predict house prices
Draw a line and find the error - use the length of the lines from the data points to the line - The error is the sum of the lengths
Keep drawing lines and calculating the error by adding the sums of the distances to find the best line.  This method of minimizing the error is called gradient descent.  In real life, we don’t want to deal with negatives so we use the least
 
**Logistic Regression** - find the line that best cuts the data - admissions office app
Number of errors - number of mis-classified points
But we don’t want to capture the number of errors.  We want to capture the log loss function
Minimize the error function - add all data points’ errors together.  Smaller is better and means there’s a better fit.
 
**Support Vector Machines** - splitting data - More with the admissions office app example - Calculate the minimum distance of the line to your points.  It’s better to have a larger minimum distance. It means you’re not cutting it too close. Maximize the distance with gradient descent.  The points close to the boundary is called support
 
**Neural Networks** - Improve admissions office example with Neural Networks - Remember the example of test scores and grades going thru multiple layers.  Yes, Yes, or 1, 1 = 1 (yes).  All other combos were 0 (no)
 
**Kernel Method** - Improve admissions office example - Another powerful method for splitting points in the plane.  Use a curve or think outside of the plane into more dimensions - This is called the Kernel Trick
 
**K Means Clustering** - best locations of 3 pizza parlors based on clientele locations
 
**Hierarchical Clustering** - pick a distance - group closest 2 houses, group next closest groups of houses, group next 2 and expand the group to include another house.  Keep going until the distance exceeds what you originally set as the max distance. Hierarchical Clustering is useful when we don’t know the number of clusters but we have an idea of how far we want them to be
 
## Lesson 3: Introductory Practice Project - See titanic notebook
 
## Lesson 5: NumPy and pandas Assessment
How would you select the last 100 rows of a 2-dimensional NumPy array ‘X'?
X[-100:,]
 
Given a Pandas dataframe 'df' with columns 'gender' and 'age', how would you compute the average age for each gender?
df.groupby(‘gender’)[‘age’].mean()
 
Which of the following commands would you use to visualize the distribution of 'height' values in a Pandas dataframe ‘df'?
df[‘height’].plot(kind=‘box’)
 
## Lesson 6: Training and Testing Models
 
Reading in data into pandas and creating a numpy array of features and labels:
```
import pandas as pd
import numpy as np
 
data = pd.read_csv("data.csv")
 
# TODO: Separate the features and the labels into arrays called X and y
 
X = np.array(data[['x1', 'x2']])
y = np.array(data[‘y'])
# import statements for the classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
 
# TODO: Pick an algorithm from the list:
# - Logistic Regression
# - Decision Trees
# - Support Vector Machines
# Define a classifier (bonus: Specify some parameters!)
# and use it to fit the data, make sure you name the variable as "classifier"
# Click on `Test Run` to see how your algorithm fit the data!
 
#classifier = LogisticRegression()
classifier = DecisionTreeClassifier()
#classifier = SVC()
classifier.fit(X,y)
```
 
## Testing your models
 
**Regression** - A regression model predicts a value.  If you have y, you can predict
what x is.  Regression returns a numeric value. 
**Classification** - A classification problem is when one wants to
determine a state (positive, negative or yes, no or cat, dog).
Classification returns a state.
 
How do we find the model that generalizes well (aka can predict a new value well
without over-fitting)
 
Testing will show us which models' errors are smaller against the test dataset
 
## Questions:
```
# Read in the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]
```
 
* "labels" are meant to be the target?
* When you split the columns of the original dataset into arrays, how does it know
which columns belong together?
 
```
# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
 
# Import the train test split
# http://scikit-learn.org/0.16/modules/generated/sklearn.cross_validation.train_test_split.html
from sklearn.cross_validation import train_test_split
 
# Read in the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]
 
# Use train test split to split your data
# Use a test size of 25% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# TODO: Create the decision tree model and assign it to the variable model.
model = DecisionTreeClassifier()
 
# TODO: Fit the model to the training data.
model.fit(X_train,y_train)
 
# TODO: Make predictions on the test data
y_pred = model.predict(X_test)
 
# TODO: Calculate the accuracy and assign it to the variable acc. on the test data
acc = accuracy_score(y_test, y_pred)
```
 
## Evaluation Metrics
### How well is my model doing?
### Confusion Matrix
 
Shows the number of true positives, false negatives, false positives, and true negatives
 
* Type 1 Error (Error of the first kind, or False Positive): In the medical example,
this is when we misdiagnose a healthy patient as sick.
* Type 2 Error (Error of the second kind, or False Negative): In the medical example,
this is when we misdiagnose a sick patient as healthy.
 
## Accuracy
 
Add up your "true" classifications and divide it by the total
 
**Accuracy** can be caclulated in sklearn by using the accuracy_score function:
```
from sklearn.metrics import accuracy_score
accuracy_score (y_true, y_pred)
```
 
## False positives and negatives
 
* **High Recall** - Medical Model example - false positives are ok (send a healthy patient for more
tests) False negatives are not ok (send a sick patient home)
 
* **High Precision** - Spam Detector example - false positives are not ok (send grandma's
email to spam folder) False negatives are ok (don't necessarily need to find all spam)
 
* Precision = column 1, row 1 of the confusion matrix (true positives) divided by
true positives plus col1, row2 of the consfusion matrix (false positives)
 
Recall - Out of all the sick patients, how many did we correctly diagnose as sick.
 
Recall = true positives/(true positives + false negatives)
 
* Harmonic Mean - F1 score
* F1 score = 2 * ((precision*recall)/(precision+recall))
 
**F-beta Score** - in the spectrum, you'd have: precision, Fbeta score, F1 score, recall
 
* F-beta score = Beta requires good intuition and knowledge of your data.  Ex: Credit card
fraud.  Can detect all fraud, but then you're sending too many fraud notifications
for non fraud transactions
 
* ROC - Area under the curve - the closer your value is to 1, the better your model is
 
## Lesson 10 - Model Selection
## Types of Errors
 
* **Underfitting** - when you oversimplify a problem - an indication is when it doesn't
do well on the training dataset.  We call this an error due to bias.
* **Overfitting** - over complicate the problem - The model is too specific.  Does well
with the training set but tends to memorize it instead of learning the characteristics. 
Does not do well on test set.  We call this an error due to variance.  The model doesn't
generalize well to the testing set
 
References picutre
 
This shows 3 types of models. 
Filled in points - train dataset
Open points - test dataset
Calculate the test number of errors and train number of errors.  Graph them.  This is called the model
complexity graph.
The left hand side is underfitted, right hand side is overfitted. 
Model in the middle is best fit.
 
* Never use test set until very very end
* You can create a train, cross validation, and test set
    * Training set is used for training the parameters
    * The cross validation set is used for making decisions about your model such as the degree of the
polynomial
    * Testing set is used for final testing of the model
 
 
**K-fold cross validation** - How can we separate our data into training and test without throwing away data points but still not cheat
and use test data for training?  Break data into multiple buckets and train model k times each time using a different bucket as our testing
and the remainging points as our training set.  Then average the error to get a final model.
 
The data set is divided into k subsets and each time, one of the k subsets is used as the test set and the other k-1 subsets are
put together to form a training set.  Then the average error across all k trials is computed.  This helps prevent overfitting. 
In sklearn:
```
from sklearn.model_selection import KFold
kf = KFold(12,3, shuffle = True)
 
for training_indices, test_indices in kf:
   print training_indices, test_indices
```
Where the parameters are the size of the data adn the size of the testing set 
Set the shuffle parameter to true to randomize the data
 
**Learning Curves** - can help you determine if a model is over-fitting, under-fitting, or just right. 
 
Pictures called learning curves
 
**Learning Curve Function**:
```
train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))
```
* estimator, is the actual classifier we're using for the data, e.g., LogisticRegression() or GradientBoostingClassifier().
* X and y is our data, split into features and labels.
* train_sizes are the sizes of the chunks of data used to draw each point in the curve.
* train_scores are the training scores for the algorithm trained on each chunk of data.
* test_scores are the testing scores for the algorithm trained on each chunk of data.
 
Two very important observations:
 
The training and testing scores come in as a list of 3 values, and this is because the function uses 3-Fold Cross-Validation.
Very important: As you can see, we defined our curves with Training and Testing Error, and this function defines
them with Training and Testing Score. These are opposite, so the higher the error, the lower the score.
Thus, when you see the curve, you need to flip it upside down in your mind, in order to compare it with the curves above.
 
```
#Import, read, and split data
import pandas as pd
data = pd.read_csv('data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])
 
# Fix random seed
np.random.seed(55)
 
### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
 
# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.
 
### Logistic Regression
#estimator = LogisticRegression()
 
### Decision Tree
#estimator = GradientBoostingClassifier()
 
### Support Vector Machine
#estimator = SVC(kernel='rbf', gamma=1000)
```
 
Code used to draw the learning curves:
```
from sklearn.model_selection import learning_curve
 
# It is good to randomize the data before drawing Learning Curves
def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation,:]
    Y2 = Y[permutation]
    return X2, Y2
 
X2, y2 = randomize(X, y)
 
def draw_learning_curves(X, y, estimator, num_trainings):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))
 
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
      plt.grid()
 
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
 
    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")
 
 
    plt.legend(loc="best")
 
    plt.show()
```
   
## Steps for training a Logistic Regression Model
* Train a bunch of models with our training data
  * The parameters on the algorithm in the example are the coeficients of the polynomial
  * The degree of the polynomial is a **meta-parameter**.  We call them **hyper-parameters**
  * For a decision tree, one hyper-parameter is depth of the tree
  * For a support vector machine, we have a hyper-parameter called the kernal
    * The **kernal** can be linear, polynomial
    * We also have Gamma (big C-looking character)
    * Pick the best combination of kernal and gamma by using grid search cross validation
* Use cross validation data to pick the best of the models
   *  Maybe calculate the F1 score and pick the model with the highest score
* Test with testing data to make sure our model is good
 
**Grid Search Cross Validation** Used to pick the best combination of hyper-parameters for a SVM
 
Insert Grid search picture
 
**How to use machine learning**
 
Insert how to use machine learning picture
 
* Start with data you wish to classify
* Use algorithms like logistic regression, neural networks, polynomial regression, SVMs, decision trees
random forests, etc
* Metrics are used to test your models, pick the best one,   like model complexity graphs,
accuracy, precision, recall, F1 score, learning curves, etc
