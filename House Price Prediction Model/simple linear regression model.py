import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Fix for NumPy print options
np.set_printoptions(threshold=np.inf)

# Import dataset
dataset = pd.read_csv(r"C:\Users\JANHAVI\Desktop\House_data.csv")
space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# Fix for sklearn train_test_split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting simple linear regression
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predictions
pred = regressor.predict(xtest)

# Visualizing training results
plt.scatter(xtrain, ytrain, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

# Visualizing test results
plt.scatter(xtest, ytest, color='red')
plt.plot(xtrain, regressor.predict(xtrain), color='blue')
plt.title("Visuals for Test Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
