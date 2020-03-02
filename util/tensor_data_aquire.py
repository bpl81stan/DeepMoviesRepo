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
import util.data_aquire as ml_data

GENRE_COLUMN = "genres"
ITEM_COLUMN = "movieId"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "titles"
USER_COLUMN = "userId"
OVERVIEW_COLUMN = 'overview'
RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"

ML_20M = "ml-20m"
DATASETS = [ML_20M]
GENRES = ['(no genres listed)',
            'Action',
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

RATING_COLUMNS = [USER_COLUMN, ITEM_COLUMN, RATING_COLUMN, TIMESTAMP_COLUMN]
MOVIE_COLUMNS = [ITEM_COLUMN, TITLE_COLUMN, GENRE_COLUMN]
OVERVIEW_COLUMNS = [ITEM_COLUMN, OVERVIEW_COLUMN]

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


def get_25M_ratings():
    path = os.path.join(ROOT_DIR, 'data', 'ml-25m')

    # read in 5. ratings
    path_data = os.path.join(path, RATINGS_FILE)
    ratings = pd.read_csv(path_data, sep=",", header=0)

    return ratings


def get_25M_movies():
    path = os.path.join(ROOT_DIR, 'data', 'ml-25m')

    # read in 5. ratings
    path_data = os.path.join(path, MOVIES_FILE)
    movies = pd.read_csv(path_data, sep=",", header=0)

    movies = ml_data.get_genres(movies)

    return movies
