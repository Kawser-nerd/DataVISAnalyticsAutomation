# import the libraries to perform Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # going to divide the given dataset in training and testing part
from sklearn.metrics import mean_squared_error, r2_score

dataFrame = pd.read_csv('insurance.csv')
#print('number of rows and columns:', dataFrame.shape)
#print(dataFrame.head()) # first five rows to see the dataset

# we want to see the distribution bmi vs charges
# we can visualize the data present in the dataframe and can visualize the relation of bmi and charge values
# we want to plot line graph
'''
sns.lmplot(x='bmi', y = 'charges', data=dataFrame, height=6)
plt.xlabel('Bmi distribution')
plt.ylabel('Charges distribution')
plt.title('BMI vs Charges')
plt.show()
'''
# to view the statistics of the given data
#print(dataFrame.describe())

# to see whether our columns/features are important/correlated with each other
# with low correlation :- Discrete data, with high correlation: Continuous data
'''
del dataFrame['sex']
del dataFrame['smoker']
del dataFrame['region']
corr =  dataFrame.corr()
# use heatmap to see the correlation intensity
sns.heatmap(corr, cmap='viridis', annot=True)
plt.show()
'''

# if you want to ensure the correlation distribution among two feature, like bmi and charges, we need to use line graph
# with distribution pattern
'''
f = plt.figure(figsize=(12,4))
ax = f.add_subplot(121)
sns.histplot(data=dataFrame['charges'], color='r', bins=50, ax=ax)
ax.set_title('charges distribution')

ax = f.add_subplot(122)
sns.histplot(dataFrame['bmi'], color='b', bins=50, ax=ax)
ax.set_title('Bmi distribution')

plt.tight_layout()
plt.show()
'''

# Machine Learning

## we can go with only the numerical column values at the beginnig
del dataFrame['sex']
del dataFrame['smoker']
del dataFrame['region']

# to train the regressed model, we need to divide the dataset in training and testing section (80%:20%)
'''
Training and testing values distribution mainly generate two parts of data.. 
X values: The values which are going to be used as input to the model
Y Values: the true values which are going to be predicted by the model
X_train will have the values being used as training and validation sets
X_test will have the values for testing the model
Y_train will have the values being used for true label at the time of training
Y_test will have the values being used for testing the model
'''
'''
X=dataFrame[['age', 'bmi', 'children']]
y=dataFrame['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# generate the model
model = LinearRegression()
model.fit(X_train, y_train) # to train the model
# evaluate the model: for this purpose we use test dataset
y_pred = model.predict(X_test) # this is the accuracy of the model with new data

# To evaluate the model we can use MSE: Mean Square Error: or We can use R square
mse = mean_squared_error(y_test, y_pred)
print(mse)

r2_score = r2_score(y_test, y_pred)
print(r2_score)

# Regressed Line visualization for the trained model

f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test, color='r', ax=ax)
sns.scatterplot(y_pred, color='b', ax=ax)
ax.set_title('Check for Liner Regression: Acutual predicted values vs actual values')

# Check for residual normality and mean
ax = f.add_subplot(122)
sns.histplot((y_test - y_pred), ax=ax, color='b')
ax.axvline((y_test - y_pred).mean(), color='k', linestyle='--')
ax.set_title('Check the residual normality & mean')
plt.tight_layout()
plt.show()
'''
'''
The current model is not trained properly, it is generating a lot of wrong results. To reduce this error, we need to 
use other columns/fetures available in the dataset

To use that, we need to embed/encode the string values to a vector representation
The model we can use for this is one-hot encode, which will transfer the string to a vector representation/number
'''

#### Step One: Data Preprocessing
# String encoding consisting of one hot encode and we will generate dumy values from it

dataFrame = pd.read_csv('insurance.csv')
categorial_columns =['sex', 'smoker', 'region']
df_encoded = pd.get_dummies(dataFrame, prefix='One', prefix_sep='_',
                            columns=categorial_columns, dtype='int8', drop_first=True)
print('Original Dataframe Shape: ', dataFrame.shape)
print('Original Dataframe Columns', dataFrame.columns)
print('Encoded Dataframe Shape: ', df_encoded.shape)
print('Encoded Dataframe Columns', df_encoded.columns)

# log transform
df_encoded['charges'] = np.log(df_encoded['charges'])
print(df_encoded['charges'])

# we will redo the training, testing and validation process with this newly process dataframe
X=df_encoded.drop('charges', axis=1)
y=df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# generate the model
model = LinearRegression()
model.fit(X_train, y_train) # to train the model
# evaluate the model: for this purpose we use test dataset
y_pred = model.predict(X_test) # this is the accuracy of the model with new data

# To evaluate the model we can use MSE: Mean Square Error: or We can use R square
mse = mean_squared_error(y_test, y_pred)
print(mse)

r2_score = r2_score(y_test, y_pred)
print(r2_score)

# Regressed Line visualization for the trained model

f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.scatterplot(y_test, color='r', ax=ax)
sns.scatterplot(y_pred, color='b', ax=ax)
ax.set_title('Check for Liner Regression: Acutual predicted values vs actual values')

# Check for residual normality and mean
ax = f.add_subplot(122)
sns.histplot((y_test - y_pred), ax=ax, color='b')
ax.axvline((y_test - y_pred).mean(), color='k', linestyle='--')
ax.set_title('Check the residual normality & mean')
plt.tight_layout()
plt.show()






