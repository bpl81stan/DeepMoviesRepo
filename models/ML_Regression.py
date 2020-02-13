import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import util.data_aquire as ml_data
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

movie_features = ml_data.create_100k_embeddings()

# print features
for x in movie_features.columns:
    print(x)



x = movie_features.drop(['user_id','item_id','rating'], axis=1)
y = movie_features['rating']

print("Length of X:")

print(len(x))

print("Length of Y:")

print(len(y))

movie_X_train, movie_X_test, movie_y_train, movie_y_test = train_test_split(x, y, test_size=0.05, random_state=42)


print("Length of X-train:")
print(movie_X_train.shape)
print("Length of Y-train:")
print(movie_y_train.shape)

print("Length of X-test:")
print(movie_X_test.shape)
print("Length of Y-test:")
print(movie_y_test.shape)





# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(movie_X_train, movie_y_train)

# Make predictions using the testing set
movie_y_pred = regr.predict(movie_X_test)

print("Length of y-pred:")
print(movie_y_pred.shape)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(movie_y_test, movie_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(movie_y_test, movie_y_pred))

print('Explained variance score : %.2f'
      % explained_variance_score(movie_y_test, movie_y_pred))


