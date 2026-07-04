import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, metrics
import numpy as np

data = pd.read_csv("weight-height.csv") # reading file data

df = pd.DataFrame(data) # creating datatabel/frame for the data
x = df[["Height"]] # setting Height column values to X and Weight to Y
y = df["Weight"]

model = linear_model.LinearRegression() # creating an empty linear model
model.fit(x, y) # training the model with X & Y values. 

y_pred = model.predict(x) # making the model look at X and predict Y and give the predictions to Y_pred variable.

print("r2 is: ", metrics.r2_score(y, y_pred)) # scoring with R2 to see accurate the predictions were.
print("RMSE is:", np.sqrt(metrics.mean_squared_error(y, y_pred))) # measuring how off the model was on average. 
plt.scatter(x, y, color="red", label="Weight Height") # creating scatter plot 
plt.legend() # so labels can be shown in the figure
plt.plot(x, y_pred, color="black", linewidth=3) # drawing the prediction line
plt.show() # figure show
