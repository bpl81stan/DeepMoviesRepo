from sklearn.model_selection import train_test_split
import util.data_aquire as ml_data
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score #, explained_variance_score, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from definitions import ROOT_DIR
import os
from collections import Counter

def run_models(start=10000, stop=100001, step=10000):

    sample_sizes = [*range(start,stop,step)]
    mse=[]
    r2=[]
    MAX_RATING = 5.0


    for n in sample_sizes:

        # movie_features = ml_data.create_100k_embeddings()
        movie_features = ml_data.create_25M_embeddings(n)

        # print features
        # for x in movie_features.columns:
        #     print(x)

        # x = movie_features.drop(['user_id','item_id','rating'], axis=1)

        x = movie_features.drop(['userId','movieId','rating'], axis=1)

        y = movie_features['rating']/MAX_RATING

        print(Counter(y))

        print("Length of X:")

        print(len(x))

        print("Length of Y:")

        print(len(y))

        movie_X_train, movie_X_test, movie_y_train, movie_y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # print("Length of X-train:")
        # print(movie_X_train.shape)
        # print("Length of Y-train:")
        # print(movie_y_train.shape)
        #
        # print("Length of X-test:")
        # print(movie_X_test.shape)
        # print("Length of Y-test:")
        # print(movie_y_test.shape)

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(movie_X_train, movie_y_train)

        # Make predictions using the testing set
        movie_y_pred = regr.predict(movie_X_test)

        print("Y Distributions:")
        print(min(movie_y_pred))
        print(max(movie_y_pred))

        print(Counter(movie_y_test))

        # print("Length of y-pred:")
        # print(movie_y_pred.shape)

        # print('Coefficients: \n', regr.coef_)

        # print('Mean Absolute Error: %.2f'
        #       % mean_absolute_error(movie_y_test, movie_y_pred))

        mse.append(mean_squared_error(movie_y_test, movie_y_pred))
        r2.append(r2_score(movie_y_test, movie_y_pred))

        # The mean squared error
        print('Mean squared error: %.4f'
              % mean_squared_error(movie_y_test, movie_y_pred))

        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.4f'
              % r2_score(movie_y_test, movie_y_pred))

        # print('Explained variance score : %.2f'
        #       % explained_variance_score(movie_y_test, movie_y_pred))

    return sample_sizes, mse, r2

def plot_models(sample_sizes, mse, r2):

    plt.plot(sample_sizes, mse, color='blue', marker=".", markersize=10)
    plt.plot(sample_sizes, r2, color='red', marker=".", markersize=10)
    plt.show()

def main():

    sample_sizes, mse, r2 = run_models(start=100000, stop=100001, step=10000)

    accuracy_metrics = pd.DataFrame(data={
                                    'sample': sample_sizes,
                                    'mse': mse,
                                    'r2': r2
                                    })

    accuracy_metrics.to_csv(os.path.join(ROOT_DIR, 'data', 'output', 'linear_regression_accuracy-2020-03-05.csv'))

#    plot_models(sample_sizes, mse, r2)


main()

