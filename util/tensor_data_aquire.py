import os
import numpy as np
import tensorflow as tf
from util import file_io
from definitions import ROOT_DIR
import functools
import pandas as pd
# from absl import app as absl_app
# from absl import flags
# from absl import logging
# import util.movielens as movielens
import util.data_aquire as ml_data
import util.text_processing as text

GENRE_COLUMN = "genres"
ITEM_COLUMN = "movieId"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "titles"
USER_COLUMN = "userId"
OVERVIEW_COLUMN = 'overview'
RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"
OVERVIEW_FILE = "movie_overview.csv"

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'ml-25m')
IMDB_PATH = os.path.join(ROOT_DIR, 'data', 'imdb')
BUFFER_PATH = os.path.join(ROOT_DIR,'buffer')


ML_25M = "ml-25m"
DATASETS = [ML_25M]
GENRES = ['Action',
            'Adventure',
            'Animation',
            'Children',
            'Comedy',
            'Crime',
            'Documentary',
            'Drama',
            'Fantasy',
            'Film-Noir',
            'Horror',
            'IMAX',
            'Musical',
            'Mystery',
            'Romance',
            'Sci-Fi',
            'Thriller',
            'War',
            'Western'
            ]
N_GENRE = len(GENRES)

DROP_COLUMNS = ['title','imdbId','tmbdId']

RATING_COLUMNS = [USER_COLUMN, ITEM_COLUMN, RATING_COLUMN, TIMESTAMP_COLUMN]
MOVIE_COLUMNS = [ITEM_COLUMN, TITLE_COLUMN, GENRE_COLUMN]
OVERVIEW_COLUMNS = [ITEM_COLUMN, OVERVIEW_COLUMN]

# Note: Users are indexed [1, k], not [0, k-1]
NUM_USER_IDS = 138493

# Note: Movies are indexed [1, k], not [0, k-1]
# Both the 1m and 20m datasets use the same movie set.
NUM_ITEM_IDS = 3952

MAX_RATING = 5

NUM_RATINGS = 20000263

NUMBER_TEXT_FEATURES = 500



_FEATURE_MAP = {
    USER_COLUMN: tf.io.FixedLenFeature([1], dtype=tf.int64),
    ITEM_COLUMN: tf.io.FixedLenFeature([1], dtype=tf.int64),
    TIMESTAMP_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.int64),
    GENRE_COLUMN: tf.io.FixedLenFeature([N_GENRE], dtype=tf.int64),
    OVERVIEW_COLUMN: tf.io.FixedLenFeature([NUMBER_TEXT_FEATURES], dtype=tf.float32),
    RATING_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.float32)
}

_BUFFER_SIZE = {"train": 2175203810, "eval": 543802008}

_USER_EMBEDDING_DIM = 16
_ITEM_EMBEDDING_DIM = 64


def get_25M_ratings():

    # read in 5. ratings
    path_data = os.path.join(DATA_PATH, RATINGS_FILE)
    ratings = pd.read_csv(path_data, sep=",", header=0)

    return ratings

def get_genres(df):

    def map_genres(current_record):

        current_record.replace("Children's", "Children")  # naming difference.
        movie_genres = current_record.split("|")
        output = np.zeros((len(GENRES),), dtype=np.int64)
        for i, genre in enumerate(GENRES):
            if genre in movie_genres:
                output[i] = 1
        return output

    df[GENRE_COLUMN] = df[GENRE_COLUMN].apply(map_genres)

    return df


def get_25M_movies():

    # read in movies

    path_data = os.path.join(DATA_PATH, MOVIES_FILE)
    movies = pd.read_csv(path_data, sep=",", header=0)

    movies = get_genres(movies)

    return movies

def get_overviews():

    def map_to_list(current_overview):
        return list(current_overview)

    # read in overviews
    path_data = os.path.join(IMDB_PATH, OVERVIEW_FILE)
    overviews = pd.read_csv(path_data, sep=",", header=0)

    overview_vectors = text.getHashVector(overviews['overview'], features=NUMBER_TEXT_FEATURES)

    overviews[OVERVIEW_COLUMN] = [np.array(x) for x in overview_vectors]

    return overviews

def construct_df(sample_size):

    ratings = get_25M_ratings()

    ratings = ratings.sample(n=sample_size)

    movies = get_25M_movies()

    overviews = get_overviews()

    movies = movies.merge(overviews, on=ITEM_COLUMN)

    movies = movies.drop(columns=DROP_COLUMNS)

    ratings = ratings.merge(movies, on=ITEM_COLUMN)

    return ratings

def construct_features():

    user_id = tf.feature_column.categorical_column_with_vocabulary_list(
        USER_COLUMN, range(1, NUM_USER_IDS))
    user_embedding = tf.feature_column.embedding_column(
        user_id, _USER_EMBEDDING_DIM, max_norm=np.sqrt(_USER_EMBEDDING_DIM))

    item_id = tf.feature_column.categorical_column_with_vocabulary_list(
        ITEM_COLUMN, range(1, NUM_ITEM_IDS))
    item_embedding = tf.feature_column.embedding_column(
        item_id, _ITEM_EMBEDDING_DIM, max_norm=np.sqrt(_ITEM_EMBEDDING_DIM))

    time = tf.feature_column.numeric_column(TIMESTAMP_COLUMN)
    genres = tf.feature_column.numeric_column(
        GENRE_COLUMN, shape=(N_GENRE,), dtype=tf.uint8)

    overviews = tf.feature_column.numeric_column(OVERVIEW_COLUMN, shape=(NUMBER_TEXT_FEATURES,))

    feature_columns = [user_embedding, item_embedding, time, genres, overviews]

    return feature_columns

def _deserialize(examples_serialized):
  features = tf.io.parse_example(examples_serialized, _FEATURE_MAP)
  return features, features[RATING_COLUMN] / MAX_RATING

def _buffer_path(data_dir, dataset, name):
  return os.path.join(data_dir, dataset,
                      "{}_{}_buffer".format(dataset, name))


def df_to_input_fn(df, name, batch_size, repeat, shuffle):
  """Serialize a dataframe and write it to a buffer file."""
  expected_size = _BUFFER_SIZE.get(name)

  buffer_write_path = _buffer_path(BUFFER_PATH, ML_25M, name)

  file_io.write_to_buffer(
      dataframe=df, buffer_path=buffer_write_path,
      columns=list(_FEATURE_MAP.keys()), expected_size=expected_size)

  def input_fn():
    dataset = tf.data.TFRecordDataset(buffer_write_path)
    # batch comes before map because map can deserialize multiple examples.
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_deserialize, num_parallel_calls=16)
    if shuffle:
      dataset = dataset.shuffle(shuffle)

    dataset = dataset.repeat(repeat)
    return dataset.prefetch(1)

  return input_fn

def construct_input_fns(batch_size=16, repeat=1, sample_size=100000):

    df = construct_df(sample_size)

    train_df = df.sample(frac=0.8, random_state=0)
    eval_df = df.drop(train_df.index)

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    train_input_fn = df_to_input_fn(df=train_df, name="train", batch_size=batch_size, repeat=repeat,shuffle=NUM_RATINGS)

    eval_input_fn = df_to_input_fn(df=eval_df, name="eval", batch_size=batch_size, repeat=repeat, shuffle=None)

    model_column_fn = functools.partial(construct_features)

    #train_input_fn()

    return train_input_fn, eval_input_fn, model_column_fn

def main():

    df = construct_df()
    print(df.columns)

    print("First Row:")
    for col in range(len(df.columns)):
        print(df.iloc[0,col])

#main()





