import os
import numpy as np
import tensorflow as tf
from util import file_io
from definitions import ROOT_DIR
import functools
import pandas as pd
from absl import app as absl_app
from absl import flags
from absl import logging
import util.movielens as movielens

GENRE_COLUMN = "genres"
ITEM_COLUMN = "item_id"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "titles"
USER_COLUMN = "user_id"

RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"

ML_20M = "ml-20m"
DATASETS = [ML_20M]

GENRES = [
    'Action', 'Adventure', 'Animation', "Children", 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', "IMAX", 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]
N_GENRE = len(GENRES)

RATING_COLUMNS = [USER_COLUMN, ITEM_COLUMN, RATING_COLUMN, TIMESTAMP_COLUMN]
MOVIE_COLUMNS = [ITEM_COLUMN, TITLE_COLUMN, GENRE_COLUMN]

# Note: Users are indexed [1, k], not [0, k-1]
NUM_USER_IDS = {
    ML_20M: 138493,
}

# Note: Movies are indexed [1, k], not [0, k-1]
# Both the 1m and 20m datasets use the same movie set.
NUM_ITEM_IDS = 3952

MAX_RATING = 5

NUM_RATINGS = {
    ML_20M: 20000263
}


_BUFFER_SUBDIR = "buffer"
_FEATURE_MAP = {
    USER_COLUMN: tf.io.FixedLenFeature([1], dtype=tf.int64),
    ITEM_COLUMN: tf.io.FixedLenFeature([1], dtype=tf.int64),
    TIMESTAMP_COLUMN: tf.io.FixedLenFeature([1],
                                                             dtype=tf.int64),
    GENRE_COLUMN: tf.io.FixedLenFeature(
        [N_GENRE], dtype=tf.int64),
    RATING_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.float32),
}

_BUFFER_SIZE = {
    movielens.ML_1M: {"train": 107978119, "eval": 26994538},
    movielens.ML_20M: {"train": 2175203810, "eval": 543802008}
}

_USER_EMBEDDING_DIM = 16
_ITEM_EMBEDDING_DIM = 64

def build_model_columns(dataset):
  """Builds a set of wide and deep feature columns."""
  user_id = tf.feature_column.categorical_column_with_vocabulary_list(
      movielens.USER_COLUMN, range(1, movielens.NUM_USER_IDS[dataset]))
  user_embedding = tf.feature_column.embedding_column(
      user_id, _USER_EMBEDDING_DIM, max_norm=np.sqrt(_USER_EMBEDDING_DIM))

  item_id = tf.feature_column.categorical_column_with_vocabulary_list(
      movielens.ITEM_COLUMN, range(1, movielens.NUM_ITEM_IDS))
  item_embedding = tf.feature_column.embedding_column(
      item_id, _ITEM_EMBEDDING_DIM, max_norm=np.sqrt(_ITEM_EMBEDDING_DIM))

  time = tf.feature_column.numeric_column(movielens.TIMESTAMP_COLUMN)
  genres = tf.feature_column.numeric_column(
      movielens.GENRE_COLUMN, shape=(movielens.N_GENRE,), dtype=tf.uint8)

  deep_columns = [user_embedding, item_embedding, time, genres]
  wide_columns = []

  return wide_columns, deep_columns


def _deserialize(examples_serialized):
  features = tf.io.parse_example(examples_serialized, _FEATURE_MAP)
  return features, features[movielens.RATING_COLUMN] / movielens.MAX_RATING


def _buffer_path(data_dir, dataset, name):
  return os.path.join(data_dir, _BUFFER_SUBDIR,
                      "{}_{}_buffer".format(dataset, name))


def _df_to_input_fn(df, name, dataset, data_dir, batch_size, repeat, shuffle):
  """Serialize a dataframe and write it to a buffer file."""
  buffer_path = _buffer_path(data_dir, dataset, name)
  expected_size = _BUFFER_SIZE[dataset].get(name)

  file_io.write_to_buffer(
      dataframe=df, buffer_path=buffer_path,
      columns=list(_FEATURE_MAP.keys()), expected_size=expected_size)

  def input_fn():
    dataset = tf.data.TFRecordDataset(buffer_path)
    # batch comes before map because map can deserialize multiple examples.
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_deserialize, num_parallel_calls=16)
    if shuffle:
      dataset = dataset.shuffle(shuffle)

    dataset = dataset.repeat(repeat)
    return dataset.prefetch(1)

  return input_fn


def _check_buffers(data_dir, dataset):
  train_path = os.path.join(data_dir, _BUFFER_SUBDIR,
                            "{}_{}_buffer".format(dataset, "train"))
  eval_path = os.path.join(data_dir, _BUFFER_SUBDIR,
                           "{}_{}_buffer".format(dataset, "eval"))

  if not tf.io.gfile.exists(train_path) or not tf.io.gfile.exists(eval_path):
    return False

  return all([
      tf.io.gfile.stat(_buffer_path(data_dir, dataset, "train")).length ==
      _BUFFER_SIZE[dataset]["train"],
      tf.io.gfile.stat(_buffer_path(data_dir, dataset, "eval")).length ==
      _BUFFER_SIZE[dataset]["eval"],
  ])


def construct_input_fns(dataset, data_dir, batch_size=16, repeat=1):
  """Construct train and test input functions, as well as the column fn."""
  if _check_buffers(data_dir, dataset):
    train_df, eval_df = None, None
  else:
    df = movielens.csv_to_joint_dataframe(dataset=dataset, data_dir=data_dir)
    df = movielens.integerize_genres(dataframe=df)
    df = df.drop(columns=[movielens.TITLE_COLUMN])

    train_df = df.sample(frac=0.8, random_state=0)
    eval_df = df.drop(train_df.index)

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

  train_input_fn = _df_to_input_fn(
      df=train_df, name="train", dataset=dataset, data_dir=data_dir,
      batch_size=batch_size, repeat=repeat,
      shuffle=movielens.NUM_RATINGS[dataset])
  eval_input_fn = _df_to_input_fn(
      df=eval_df, name="eval", dataset=dataset, data_dir=data_dir,
      batch_size=batch_size, repeat=repeat, shuffle=None)
  model_column_fn = functools.partial(build_model_columns, dataset=dataset)

  train_input_fn()
  return train_input_fn, eval_input_fn, model_column_fn

def main(_):
#  movielens.download(dataset=flags.FLAGS.dataset, data_dir=flags.FLAGS.data_dir)

    path = os.path.join(ROOT_DIR, 'data', 'ml-25m')
    dataset = 'ml-25m'

    construct_input_fns(dataset, path)

if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  #define_data_download_flags()
  #flags.adopt_module_key_flags(movielens)
  #flags_core.set_defaults(dataset="ml-1m")
  absl_app.run(main)
