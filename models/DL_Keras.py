import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

from definitions import ROOT_DIR
import util.tensor_data_aquire as ml_data
import pandas as pd
import numpy as np
from tensorflow.keras import metrics
from collections import Counter

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

# class weights
CLASS_1_WEIGHT=1.599481040751487
CLASS_0_WEIGHT=2.6681094680599697
CLASS_WEIGHTS={0:CLASS_0_WEIGHT, 1:CLASS_1_WEIGHT}

def run_models(sample_size, model_name, epochs=5, batch_size=256, learning_rate=.001):

    TRAIN_EPOCHS=epochs
    EPOCHS_BETWEEN_EVALS=10
    INTER_OP_PARALLELISM_THREADS=0
    INTRA_OP_PARALLELISM_THREADS=0
    BATCH_SIZE=batch_size
    # HIDDEN_UNITS = [256, 256, 256, 128]

    # accuracy_df = pd.DataFrame(columns=['sample', 'epoch', 'accuracy', 'loss'])

    CURRENT_MODEL_NAME=model_name
    SAVE_MODEL_PATH = os.path.join(ROOT_DIR, 'SAVED_MODELS',CURRENT_MODEL_NAME)
    SAVE_MODEL_SAMPLES_PATH = os.path.join(ROOT_DIR, 'SAVED_MODELS',CURRENT_MODEL_NAME, str(sample_size))

    if not(os.path.exists(SAVE_MODEL_PATH)):
          os.mkdir(SAVE_MODEL_PATH)

    if not(os.path.exists(SAVE_MODEL_SAMPLES_PATH)):
          os.mkdir(SAVE_MODEL_SAMPLES_PATH)

    train_ds, val_ds, test_ds, predict_ds, predict_df, user_map, movie_map, feature_columns, train_df, val_df = \
        ml_data.get_movielens_df(val_split=.1, test_split=.2, sample_size=sample_size, batch_size=BATCH_SIZE)


    feature_layer = layers.DenseFeatures(feature_columns)

    model = tf.keras.Sequential(
        [
            feature_layer,
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            # layers.Dense(256, activation='relu'),
            # layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(1)
        ]
    )
    Adamax = tf.keras.optimizers.Adamax(
                    learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'
                    )

    # MULTI-CLASSIFIER
    model.compile(
        optimizer='adam',
        # loss=tf.keras.losses.MeanSquaredError(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        # metrics=['mse']
        metrics=['accuracy']
    )

    # BINARY CLASSIFIER
    # model.compile(
    #     optimizer=Adamax,
    #     # loss=tf.keras.losses.MeanSquaredError(),
    #     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #     # metrics=['mse']
    #     metrics=['accuracy']
    # )

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=TRAIN_EPOCHS,
              class_weight=CLASS_WEIGHTS
              )

    loss, accuracy = model.evaluate(test_ds)

    #model.summary()

    # accuracy_df = accuracy_df.append({'sample':sample_size,'epoch':TRAIN_EPOCHS,'accuracy':accuracy, 'loss':loss}, ignore_index=True)

    template = 'Sample:{}, Epoch:{}, Learning Rate{},Accuracy:{}, Loss:{}'
    print(template.format(sample_size, TRAIN_EPOCHS, learning_rate,accuracy, loss))

    # accuracy_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
    #                            'classifiers','ACCURACY - '+model_name+'.csv'))

    # y_pred = model.predict_classes(predict_ds, batch_size=None)
    #
    # y_pred = pd.DataFrame(y_pred)
    #
    # print(Counter(y_pred))

    #predict_df = predict_df.merge(y_pred, left_index=True, right_index=True)

    # predict_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
    #                            'classifiers','predict','predict_df-'+model_name+'.csv'))
    #
    # y_pred.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
    #                            'classifiers','predict','y_pred-'+model_name+'.csv'))

    # train_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
    #                            'classifiers','predict','train-'+model_name+'.csv'))

    # val_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
    #                            'classifiers','predict','val_df-'+model_name+'.csv'))

    # ROSETTA_PATH=os.path.join(SAVE_MODEL_SAMPLES_PATH, 'rosettas')
    #
    # if not(os.path.exists(ROSETTA_PATH)):
    #       os.mkdir(ROSETTA_PATH)
    #
    # pd.DataFrame.from_dict(user_map, orient='index').to_csv(
    #       os.path.join(ROSETTA_PATH, 'sample-'+str(sample_size)+'-usermap.csv')
    # )
    #
    # pd.DataFrame.from_dict(movie_map, orient='index').to_csv(
    #       os.path.join(ROSETTA_PATH, 'sample-'+str(sample_size)+'-moviemap.csv')
    # )

    return loss, accuracy


def main():

    # start = .0001
    # stop = .01
    # step = .0001
    #
    # multiplier = 10000
    #
    # start_r = int(start * multiplier)
    # stop_r = int(stop * multiplier)
    # step_r = int(step * multiplier)
    #
    # learning_rate = [*range(start_r, stop_r, step_r)]
    #
    # for i in learning_rate:
    #     print(i / multiplier)

    #learning_rate = [.1, .01, .001, .0001, .00001, .000001, .000001]
    learning_rate = [
        0.00002,
        0.0001,
        0.00015,
        0.00005,
        0.00003,
        0.00014,
        0.00018,
        0.00017,
        0.00001,
        0.0002,
        0.0015,
        0.0017,
        0.0002,
        0.0001,
        0.0009,
        0.0004,
        0.0018,
        0.0006,
        0.0014,
        0.0003
    ]

    TRAIN_EPOCHS = 5


    samples = [10000]

    accuracy_df = pd.DataFrame(columns=['sample', 'epoch', 'learning_rate' , 'accuracy', 'loss'])

    model_name = 'DL-Keras-LR_Random-2020-03-11'

    for s in samples:
        for lr in learning_rate:
            # cur_lr = lr/multiplier
            loss, accuracy = run_models(sample_size=s, model_name=model_name, learning_rate=cur_lr)

            accuracy_df = accuracy_df.append(
                {'sample': s, 'epoch': TRAIN_EPOCHS, 'learning_rate':learning_rate,'accuracy': accuracy, 'loss': loss}, ignore_index=True)


    accuracy_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
                                            'classifiers', 'ACCURACY - ' + model_name + '.csv'))

main()
