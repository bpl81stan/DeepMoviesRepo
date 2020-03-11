import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.estimator import DNNClassifier
from tensorflow import feature_column

from definitions import ROOT_DIR
import util.tensor_data_aquire as ml_data
import pandas as pd
import numpy as np

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)

def run_models(start=10000, stop=100001, step=10000):

      sample_sizes = [*range(start, stop, step)]

      mse_df = pd.DataFrame(columns=['sample','epoch', 'accuracy'])

      for n in sample_sizes:
            print("New Sample")
            print("N: " + str(n))
            # Configure training sets
            TRAIN_EPOCHS=50
            EPOCHS_BETWEEN_EVALS=10
            INTER_OP_PARALLELISM_THREADS=0
            INTRA_OP_PARALLELISM_THREADS=0
            BATCH_SIZE=256
            HIDDEN_UNITS = [256, 256, 256, 128]

            CURRENT_MODEL_NAME='DL-Classifier-Adam-500k'
            SAVE_MODEL_PATH = os.path.join(ROOT_DIR, 'SAVED_MODELS',CURRENT_MODEL_NAME)
            SAVE_MODEL_SAMPLES_PATH = os.path.join(ROOT_DIR, 'SAVED_MODELS',CURRENT_MODEL_NAME, str(n))

            if not(os.path.exists(SAVE_MODEL_PATH)):
                  os.mkdir(SAVE_MODEL_PATH)

            if not(os.path.exists(SAVE_MODEL_SAMPLES_PATH)):
                  os.mkdir(SAVE_MODEL_SAMPLES_PATH)



            train_input_fn, eval_input_fn, model_column_fn, user_map, movie_map = ml_data.construct_input_fns(
                                                                              batch_size=BATCH_SIZE,
                                                                              repeat=EPOCHS_BETWEEN_EVALS,
                                                                              sample_size=n,
                                                                              validation=False
                                                                        )

            feature_columns = model_column_fn()
            # def cust_auc(labels, predictions):
            #       # prediction arguments for DNNClassifier:
            #       # 'logits'
            #       # 'probabilities'
            #       # 'class_ids'
            #       # 'classes'
            #       # 'all_class_ids'
            #       # 'all_classes'
            #       auc_metric = tf.keras.metrics.AUC(name='cust_auc')
            #       auc_metric.update_state(y_true=tf.reshape(tf.one_hot(labels, depth=10), predictions['probabilities'].shape)
            #                               ,y_pred=predictions['probabilities'])
            #       return {'auc': auc_metric}
            #
            # def cust_accuracy(labels, predictions):
            #       # prediction arguments for DNNClassifier:
            #       # 'logits'
            #       # 'probabilities'
            #       # 'class_ids'
            #       # 'classes'
            #       # 'all_class_ids'
            #       # 'all_classes'
            #
            #       labels_one_hot=tf.one_hot(labels, depth=10)
            #       labels_one_hot=tf.reshape(labels_one_hot, [None,10])
            #       accuracy_metric = tf.keras.metrics.CategoricalCrossentropy(name='cust_accuracy')
            #       accuracy_metric.update_state(y_true=tf.reshape(tf.one_hot(labels, depth=10), predictions['probabilities'].shape)
            #                                     ,y_pred=predictions['probabilities'])
            #       return {'accuracy': accuracy_metric}

            model = DNNClassifier(
                  model_dir=SAVE_MODEL_SAMPLES_PATH,
                  feature_columns=feature_columns,
                  hidden_units=HIDDEN_UNITS,
                  n_classes=10,
                  #label_vocabulary=["0.5","1","1.5","2","2.5","3","3.5","4","4.5","5"],
                  optimizer='Adam',
                  #lambda: tf.keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
                  #'Adam',

                  # tf.keras.optimizers.Adam(
                  #                     learning_rate=tf.compat.v1.train.exponential_decay(
                  #                         learning_rate=0.001,
                  #                         global_step=tf.compat.v1.train.get_global_step(),
                  #                         decay_steps=10000,
                  #                         decay_rate=0.96)),


                  #'Adam', #tf.train.AdamOptimizer(),
                  activation_fn=tf.nn.relu,
                  dropout=0.3,
                  batch_norm=True
                  #loss_reduction=tf.losses.Reduction.MEAN
            )

            # model = tf.estimator.add_metrics(model, cust_auc)
            #
            # model = tf.estimator.add_metrics(model, cust_accuracy)

            for i in range(TRAIN_EPOCHS // EPOCHS_BETWEEN_EVALS):

                  epoch_num = (i+1)*EPOCHS_BETWEEN_EVALS

                  model.train(input_fn=train_input_fn)  # , hooks=train_hooks)

                  eval = model.evaluate(input_fn=eval_input_fn)

                  # predict = model.predict(input_fn=predict_input_fn)

                  cur_accuracy = eval['accuracy']

                  mse_df = mse_df.append({'sample':n,'epoch':epoch_num,'accuracy':eval['accuracy']}, ignore_index=True)

                  template = 'Sample:{}, Epoch:{} of {}, Accuracy:{}'
                  print(template.format(n, epoch_num, TRAIN_EPOCHS,cur_accuracy))

            mse_df = mse_df.append({'sample':n,'epoch':'final', 'accuracy':eval['accuracy']}, ignore_index=True)

            mse_df.to_csv(os.path.join(ROOT_DIR, 'data', 'output',
                                       'classifiers','DNN_Classifier_accuracy-10K-500K-2020-03-07.csv'))

            ROSETTA_PATH=os.path.join(SAVE_MODEL_SAMPLES_PATH, 'rosettas')

            if not(os.path.exists(ROSETTA_PATH)):
                  os.mkdir(ROSETTA_PATH)

            pd.DataFrame.from_dict(user_map, orient='index').to_csv(
                  os.path.join(ROSETTA_PATH, 'sample-'+str(n)+'-usermap.csv')
            )

            pd.DataFrame.from_dict(movie_map, orient='index').to_csv(
                  os.path.join(ROSETTA_PATH, 'sample-'+str(n)+'-moviemap.csv')
            )


def main():
      run_models(start=10000, stop=500001, step=10000)

main()
