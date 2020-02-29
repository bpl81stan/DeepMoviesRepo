import pandas as pd
import numpy as np
from collections import defaultdict
from definitions import ROOT_DIR
import os
import tensorflow as tf
import util.data_aquire as ml_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score #, explained_variance_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def run_model_dl():
    # movie_features = ml_data.create_100k_embeddings()
    movie_features = ml_data.create_25M_embeddings(n=10000)

    movie_features['rating'] = movie_features['rating'] * 2  # scale by 2 to factor for .5 ratings

    x = movie_features.drop(['userId', 'movieId', 'rating'], axis=1)

    y = movie_features['rating']

    movie_X_train, movie_X_test, movie_y_train, movie_y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((movie_X_train.values, movie_y_train.values))

    test_dataset = tf.data.Dataset.from_tensor_slices((movie_X_test.values, movie_y_test.values))

    print(movie_X_train.shape)


    model = tf.keras.Sequential(
        [
            #layers.Dense(64),
            layers.LSTM(10)
            #layers.Dense(1,activation='softmax')
        ])

    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])


    model.fit(train_dataset, epochs=10, verbose=0)

    movie_y_pred = model.predict(test_dataset).flatten()
    print(movie_y_pred)

    print(movie_X_test.values)

    print(movie_y_test.values)

    loss, mse = model.evaluate(test_dataset)

    return mse


def plot_models(sample_sizes, mse):

    plt.plot(sample_sizes, mse, color='blue', marker=".", markersize=10)
    # plt.plot(sample_sizes, r2, color='red', marker=".", markersize=10)
    # plt.plot(sample_sizes, r2, color='green', marker=".", markersize=10)
    plt.show()

def main():

    mse= run_model_dl()

    print(mse)

    # accuracy_metrics = pd.DataFrame(data={
    #                                     'sample': sample_sizes,
    #                                     'rmse': rmse
    #                                 })
    #
    # accuracy_metrics.to_csv(os.path.join(ROOT_DIR, 'data', 'output', 'dl_model1_accuracy.csv'))
    #
    # plot_models(sample_sizes, rmse)


main()



def run_models_lr(start=10000, stop=100001, step=10000):

    sample_sizes = [*range(start,stop,step)]
    mse_list=[]
    r2_list=[]
    # accuracy=[]
    # precision=[]
    # recall=[]

    for n in sample_sizes:

        #movie_features = ml_data.create_100k_embeddings()
        movie_features = ml_data.create_25M_embeddings(n)

        movie_features['rating']=movie_features['rating']*2 #scale by 2 to factor for .5 ratings

        x = movie_features.drop(['userId','movieId','rating'], axis=1)

        y = movie_features['rating']

        movie_X_train, movie_X_test, movie_y_train, movie_y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        train_dataset = tf.data.Dataset.from_tensor_slices((movie_X_train.values, movie_y_train.values))

        test_dataset = tf.data.Dataset.from_tensor_slices((movie_X_test.values, movie_y_test.values))

        print(movie_X_train.shape)


        model = tf.keras.Sequential(
            [
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='sigmoid'),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dense(1,activation='softmax')
            ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])


        model.fit(train_dataset, epochs=10, verbose=0)

        movie_y_pred = model.predict(test_dataset).flatten()
        print(movie_y_pred)

        print(movie_X_test.values)

        print(movie_y_test.values)

        loss, mae, mse = model.evaluate(test_dataset)

        mse_list.append(np.sqrt(mse))


        # mse.append(mean_squared_error(movie_y_test, movie_y_pred))
        # r2.append(r2_score(movie_y_test, movie_y_pred))
        # accuracy.append(accuracy_score(movie_y_test, movie_y_pred))
        #precision.append(precision_score(movie_y_test, movie_y_pred))
        #recall.append(recall_score(movie_y_test, movie_y_pred))

        print("Sample: "+ str(n))

    return sample_sizes, mse_list#, r2_list#, accuracy#, precision, recall