import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers

from definitions import ROOT_DIR
import util.tensor_data_aquire as ml_data
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger
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

def run_models(sample_size, model_name, hidden_layers, epochs=5, batch_size=256, learning_rate=.001):

    TRAIN_EPOCHS=epochs
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

    train_ds, val_ds, test_ds, predict_ds, predict_df, user_map, movie_map, \
    feature_columns, feature_layer_inputs, train_df, val_df = ml_data.get_movielens_df(batch_size=BATCH_SIZE)

    feature_layer = layers.DenseFeatures(feature_columns)

    feature_layer_outputs = feature_layer(feature_layer_inputs)
    i=0
    for l in hidden_layers:
        if i==0:
            dense_tensor = layers.Dense(l, activation='relu', name='layer' + str(i))(feature_layer_outputs)
            dense_tensor = layers.BatchNormalization()(dense_tensor)
            dense_tensor = layers.Dropout(.5)(dense_tensor)
        elif i==len(hidden_layers)-1:
            dense_tensor = layers.Dense(l, use_bias=False, name='layer' + str(i))(dense_tensor)
            # dense_tensor = layers.Dropout(.5)(dense_tensor)
        else:
            dense_tensor = layers.Dense(l, name='layer' + str(i))(dense_tensor)
            # dense_tensor = layers.Dropout(.5)(dense_tensor)
        i=i+1


    dense_tensor = layers.BatchNormalization()(dense_tensor)
    dense_tensor = layers.Dropout(.5)(dense_tensor)
    outputs = layers.Dense(1, activation='sigmoid', name='predictions')(dense_tensor)

    model = tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=outputs)

    csv_logger = CSVLogger(os.path.join(SAVE_MODEL_SAMPLES_PATH, model_name + '- training.log'))

    Adamax = tf.keras.optimizers.Adamax(learning_rate=learning_rate, beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-07, name='Adamax')

    # BINARY CLASSIFIER
    model.compile(
        optimizer=Adamax,
        #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #metrics=['categorical_accuracy']
        metrics=['accuracy', tf.keras.metrics.binary_accuracy]
    )

    # MULTI-CLASSIFIER
    # model.compile(
    #     optimizer=Adamax,
    #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=['categorical_accuracy']
    # )

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=TRAIN_EPOCHS,
              class_weight=CLASS_WEIGHTS,
              use_multiprocessing=True,
              callbacks=[csv_logger]
              )

    loss, accuracy, binary_accuracy = model.evaluate(test_ds)

    model.summary()

    # accuracy_df = accuracy_df.append({'sample':sample_size,'epoch':TRAIN_EPOCHS,'accuracy':accuracy, 'loss':loss}, ignore_index=True)

    template = 'Sample:{}, Epoch:{}, Architecture: {},Learning Rate{},Accuracy:{},Binary_Accuracy:{}, Loss:{}'
    print(template.format(sample_size, TRAIN_EPOCHS, str(hidden_layers), learning_rate,accuracy, binary_accuracy, loss))

    # accuracy_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
    #                            'classifiers','ACCURACY - '+model_name+'.csv'))

    y_pred = model.predict(predict_ds, batch_size=None)

    print(y_pred)

    y_pred = pd.DataFrame(data=y_pred, columns=['y_pred'], dtype=np.float64)
    # y_pred = y_pred.astype('float64')
    # y_pred.columns = ['y_pred']

    y_pred_binary=pd.DataFrame(data=[0 if y < 0.5 else 1 for y in y_pred['y_pred']], columns=['y_pred_binary'], dtype=np.float64)

    print(y_pred_binary)

    y_pred_count = y_pred_binary.iloc[:, 0].value_counts()

    print("Count of y_pred:")
    print(y_pred_count)

    labels = predict_df['rating'].apply(lambda x: 1 if x > 3 else 0)
    print("Count of y_actuals:")
    print(labels.value_counts)

    predict_df = predict_df.merge(y_pred, left_index=True, right_index=True)
    predict_df = predict_df.merge(y_pred_binary, left_index=True, right_index=True)

    predict_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
                               'classifiers','predict','predict_df-'+model_name+'-lr-'+str(learning_rate)+'.csv'))

    y_pred.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
                               'classifiers','predict','y_pred-'+model_name+'-lr-'+str(learning_rate)+'.csv'))

    train_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
                               'classifiers','predict','train-'+model_name+'.csv'))

    val_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
                               'classifiers','predict','val_df-'+model_name+'.csv'))

    ROSETTA_PATH=os.path.join(SAVE_MODEL_SAMPLES_PATH, 'rosettas')

    if not(os.path.exists(ROSETTA_PATH)):
          os.mkdir(ROSETTA_PATH)

    pd.DataFrame.from_dict(user_map, orient='index').to_csv(
          os.path.join(ROSETTA_PATH, 'sample-'+str(sample_size)+'-usermap.csv')
    )

    pd.DataFrame.from_dict(movie_map, orient='index').to_csv(
          os.path.join(ROSETTA_PATH, 'sample-'+str(sample_size)+'-moviemap.csv')
    )

    model_json = model.to_json()

    with open(os.path.join(SAVE_MODEL_PATH,model_name+'.json'), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(os.path.join(SAVE_MODEL_PATH,model_name+'-weights.h5'))

    model.save(os.path.join(SAVE_MODEL_PATH,model_name+'-model.h5'))

    tf.keras.backend.clear_session()

    return loss, accuracy, y_pred_count.get(0,0), y_pred_count.get(1,0)


def main():

    TRAIN_EPOCHS = 10

    learning_rate = [
        # .1,
        .01,
        # .001,
        # .0001,
        .00001,
        .000001
    ]

    samples = [1000000]

    hidden_layers=[
        # [8192, 64],
        [128,4],
        [256,256,256,128]
        # [32,512,8],
        # [4096,16,2048,512],
        # [256,4]
    ]

    arch = 1
    model_name = "blank.csv"
    today = str(datetime.date.today())
    model_subject = 'Improved-Arch-LargeM'

    accuracy_df = pd.DataFrame(
        columns=['sample', 'hidden_layers', 'epoch', 'learning_rate', 'accuracy', 'loss', 'count_0', 'count_1'])

    for hidden_layer in hidden_layers:
        for s in samples:
            for lr in learning_rate:
                # cur_lr = lr/multiplier

                model_name = today+'-'+\
                             model_subject + '-' \
                             'Arch-'+ str(arch) + '-' + \
                             'Sample-'+ str(s) + '-' + \
                             'lr-'+ str(lr)

                loss, accuracy, count_0, count_1 = run_models(sample_size=s, model_name=model_name, hidden_layers=hidden_layer, learning_rate=lr)

                accuracy_df = accuracy_df.append(
                    {'sample': s, 'epoch': TRAIN_EPOCHS, 'hidden_layers': str(hidden_layer),
                     'learning_rate':lr,'accuracy': accuracy, 'loss': loss,
                     'count_0':count_0, 'count_1':count_1
                     }, ignore_index=True)

        arch += 1

    csv_name = today + '-' + \
                model_subject

    accuracy_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
                                        'classifiers', 'ACCURACY - '+csv_name+'.csv'))




main()
