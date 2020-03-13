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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

GENRE_COLUMN = "genres"
ITEM_COLUMN = "movieId"  # movies
RATING_COLUMN = "rating"
TIMESTAMP_COLUMN = "timestamp"
TITLE_COLUMN = "title"
USER_COLUMN = "userId"
OVERVIEW_COLUMN = 'overview'

#movies_metadata.csv columns
IMDB_COLUMN = 'imdb_id'
POPULARITY_COLUMN = 'popularity'
REVENUE_COLUMN = 'revenue'
RUNTIME_COLUMN = 'runtime'
VOTE_AVERAGE_COLUMN = 'vote_average'
VOTE_COUNT_COLUMN = 'vote_count'

LINKS_IMDB_COLUMN = 'imdbId'

RATINGS_FILE = "ratings.csv"
MOVIES_FILE = "movies.csv"
OVERVIEW_FILE = "movie_overview.csv"
MOVIE_META_FILE = "movies_metadata.csv"




DATA_PATH = os.path.join(ROOT_DIR, 'data', 'ml-25m')
IMDB_PATH = os.path.join(ROOT_DIR, 'data', 'imdb')
KAGGLE_PATH = os.path.join(ROOT_DIR, 'data', 'kaggle')
BUFFER_PATH = os.path.join(ROOT_DIR,'buffer')

META_COLUMNS = ['imdb_id','popularity','revenue', 'runtime', 'vote_average', 'vote_count']

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
# NUM_USER_IDS = 138493

# Note: Movies are indexed [1, k], not [0, k-1]
# Both the 1m and 20m datasets use the same movie set.
# NUM_ITEM_IDS = 3952

MAX_RATING = 0.5

# NUM_RATINGS = 20000263

NUMBER_TEXT_FEATURES = 500

_FEATURE_MAP = {
    USER_COLUMN: tf.io.FixedLenFeature([1], dtype=tf.int64),
    ITEM_COLUMN: tf.io.FixedLenFeature([1], dtype=tf.int64),
    TIMESTAMP_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.int64),
    GENRE_COLUMN: tf.io.FixedLenFeature([N_GENRE], dtype=tf.int64),
    OVERVIEW_COLUMN: tf.io.FixedLenFeature([NUMBER_TEXT_FEATURES], dtype=tf.float32),
    POPULARITY_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.int64),
    REVENUE_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.int64),
    RUNTIME_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.int64),
    VOTE_AVERAGE_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.int64),
    VOTE_COUNT_COLUMN: tf.io.FixedLenFeature([1],dtype=tf.int64),
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

def get_movie_metadata():

    # read in movies meta data
    DTYPE_SET = lambda x: pd.to_numeric(x)
    PARSE_ID = lambda i: str(i[2:])

    path_data = os.path.join(KAGGLE_PATH, MOVIE_META_FILE)

    movies_meta = pd.read_csv(path_data, sep=",", header=0, usecols=META_COLUMNS,
                              converters={IMDB_COLUMN:PARSE_ID,
                                            POPULARITY_COLUMN:DTYPE_SET,
                                            REVENUE_COLUMN:DTYPE_SET,
                                            RUNTIME_COLUMN:DTYPE_SET,
                                            VOTE_AVERAGE_COLUMN:DTYPE_SET,
                                            VOTE_COUNT_COLUMN:DTYPE_SET}
                              )

    links = get_links()

    movies_meta = movies_meta.merge(links, how='left',left_on=IMDB_COLUMN, right_on=LINKS_IMDB_COLUMN)

    return movies_meta

def get_links():

    # read in 3. links
    path_data = os.path.join(DATA_PATH, 'links.csv')
    links = pd.read_csv(path_data, sep=",", header=0, usecols=['movieId', 'imdbId'],
                        dtype={'movieId': np.int,'imdbId': np.str})

    return links

def get_25M_movies():

    # read in movies

    path_data = os.path.join(DATA_PATH, MOVIES_FILE)
    movies = pd.read_csv(path_data, sep=",", header=0)

    movies = get_genres(movies)

    movies = movies.drop(columns=[TITLE_COLUMN], axis=1)

    return movies

def get_overviews():

    def map_to_list(current_overview):
        return list(current_overview)

    # read in overviews
    path_data = os.path.join(IMDB_PATH, OVERVIEW_FILE)
    overviews = pd.read_csv(path_data, sep=",", header=0)

    overview_vectors = text.getCountVector(overviews['overview'], features=NUMBER_TEXT_FEATURES)

    #text.getHashVector(overviews['overview'], features=NUMBER_TEXT_FEATURES)

    overviews[OVERVIEW_COLUMN] = [np.array(x) for x in overview_vectors]

    return overviews

def get_movie_overviews():

    if not(os.path.exists(os.path.join(IMDB_PATH, 'movies_data.csv'))):
        movies = get_25M_movies()

        overviews = get_overviews()

        movies = movies.merge(overviews, on=ITEM_COLUMN)

        movies_meta = get_movie_metadata()

        movies = movies.merge(movies_meta, how='left', left_on=ITEM_COLUMN, right_on=ITEM_COLUMN)

        movies = movies.drop(['imdbId_x', 'tmbdId', 'imdb_id', 'imdbId_y'], axis=1)

        movies = movies.fillna(0)

        movies.to_csv(os.path.join(IMDB_PATH, 'movies_data.csv'), index=False, index_label=False)

    else:

        ARRAY_CONVERTER = lambda x: np.fromstring(x[1:-1], sep=' ')
        movies = pd.read_csv(os.path.join(IMDB_PATH, 'movies_data.csv'),
                             converters={GENRE_COLUMN: ARRAY_CONVERTER, OVERVIEW_COLUMN: ARRAY_CONVERTER})

        # movies = movies.drop(OVERVIEW_COLUMN, axis=1)
        #
        # movies = movies.drop(GENRE_COLUMN, axis=1)

        #movies[GENRE_COLUMN] = [np.array(x) for x in movies[GENRE_COLUMN]]

        # movies[OVERVIEW_COLUMN] = [np.array(x) for x in movies[OVERVIEW_COLUMN]]

        movies = movies.fillna(0)

        #movies.astype(np.float64)

    return movies

def rebaseline_dataset(df):

    # users
    current_user_list = df[USER_COLUMN].unique()

    num_users = len(current_user_list)

    user_map = {}

    for i in range(num_users):
        user_map.update({current_user_list[i]:i+1})

    # movies
    current_movie_list = df[ITEM_COLUMN].unique()

    num_movies = len(current_movie_list)

    movie_map = {}

    for i in range(num_movies):
        movie_map.update({current_movie_list[i]: i + 1})

    new_user_index = []
    new_movie_index = []

    for i, r in df.iterrows():
        new_user_index.append(user_map.get(r[USER_COLUMN]))
        new_movie_index.append(movie_map.get(r[ITEM_COLUMN]))

    df[USER_COLUMN] = new_user_index
    df[ITEM_COLUMN] = new_movie_index

    # RESET NUMBER OF USERS, ITEMS AND RATINGS BASED ON SAMPLE DATAFRAME
    # NUM_USER_IDS = num_users
    #
    # NUM_ITEM_IDS = num_movies

    # NUM_RATINGS = len(df)

    return df, user_map, movie_map


def construct_df(sample_size):

    ratings = get_25M_ratings()

    ratings = ratings.sample(n=sample_size)

    movies = get_movie_overviews()

    ratings = ratings.merge(movies, on=ITEM_COLUMN)

    #ORIGINAL_SAMPLE = ratings

    return ratings

def construct_unrated(USER):

    movies = get_movie_overviews()


##################################################################
########## KERAS DATA AQUISITION FUNCTIONS
##################################################################

def convert_df_to_tf_ds(df, shuffle=True, batch_size=256, predict=False):

    df = df.copy()
#    df=df[[DUMMY_COLUMN, RATING_COLUMN]]

    if(predict==False):

        labels = df.pop(RATING_COLUMN)

        # labels = labels*2
        # encoder = LabelEncoder()
        # encoder.fit(labels)
        # encoded_labels = encoder.transform(labels)
        # labels_one_hot = tf.keras.utils.to_categorical(encoded_labels)

        # print("Pre - Binary Coding:")
        # print(labels)

        labels = labels.apply(lambda x: 1 if x > 3 else 0)

        # print("Binary Coding:")
        # print(labels)
        # print(labels_one_hot)

        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    else:
        df = df.drop(columns=RATING_COLUMN, axis=1)

        ds = tf.data.Dataset.from_tensor_slices((dict(df)))

    if shuffle:
            ds = ds.shuffle(buffer_size=len(df))

    ds = ds.batch(batch_size)

    return ds

def construct_movielens_df(val_split=.05, test_split=.05, sample_size=10000):

    original_dataset = construct_df(sample_size)
    #new_dataset = original_dataset
    new_dataset, user_map, movie_map = rebaseline_dataset(original_dataset)

    train_df, test_df = train_test_split(new_dataset, test_size=test_split)
    train_df, val_df = train_test_split(train_df, test_size=val_split)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df, val_df, test_df, user_map, movie_map


def get_movielens_df(batch_size=256):

    ARRAY_CONVERTER = lambda x: np.fromstring(x[1:-1], sep=' ')

    # movies = movies.drop(OVERVIEW_COLUMN, axis=1)
    #
    # movies = movies.drop(GENRE_COLUMN, axis=1)

    # movies[GENRE_COLUMN] = [np.array(x) for x in movies[GENRE_COLUMN]]

    # movies[OVERVIEW_COLUMN] = [np.array(x) for x in movies[OVERVIEW_COLUMN]]

    dataset_name = 'c-1m'

    DATASET_PATH = os.path.join(ROOT_DIR, 'data','custom_datasets', dataset_name)

    train_df = pd.read_csv(os.path.join(DATASET_PATH, 'train_df.csv'),
                           converters={GENRE_COLUMN: ARRAY_CONVERTER, OVERVIEW_COLUMN: ARRAY_CONVERTER}).fillna(0)
    val_df = pd.read_csv(os.path.join(DATASET_PATH, 'val_df.csv'),
                         converters={GENRE_COLUMN: ARRAY_CONVERTER, OVERVIEW_COLUMN: ARRAY_CONVERTER}).fillna(0)
    test_df = pd.read_csv(os.path.join(DATASET_PATH, 'test_df.csv'),
                          converters={GENRE_COLUMN: ARRAY_CONVERTER, OVERVIEW_COLUMN: ARRAY_CONVERTER}).fillna(0)
    predict_df = pd.read_csv(os.path.join(DATASET_PATH, 'predict_df.csv'),
                             converters={GENRE_COLUMN: ARRAY_CONVERTER, OVERVIEW_COLUMN: ARRAY_CONVERTER}).fillna(0)
    user_map = pd.read_csv(os.path.join(DATASET_PATH, 'usermap.csv'))
    movie_map = pd.read_csv(os.path.join(DATASET_PATH, 'moviemap.csv'))

    num_user_ids = len(user_map)
    num_item_ids = len(movie_map)

    features, feature_layer_inputs = construct_features(num_user_ids, num_item_ids)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    predict_df = predict_df.reset_index(drop=True)

    train_ds = convert_df_to_tf_ds(train_df, shuffle=True, batch_size=batch_size)
    val_ds = convert_df_to_tf_ds(val_df, shuffle=False, batch_size=batch_size)
    test_ds = convert_df_to_tf_ds(test_df, shuffle=False, batch_size=batch_size)
    predict_ds = convert_df_to_tf_ds(test_df, shuffle=False, batch_size=batch_size, predict=True)

    return train_ds, val_ds, test_ds, predict_ds, test_df, user_map, \
               movie_map, features, feature_layer_inputs, train_df, val_df


##################################################################
########## TENSORFLOW DATA AQUISITION FUNCTIONS
##################################################################


def construct_features(num_user_ids, num_item_ids):

    feature_columns = []
    feature_layer_inputs = {}

    # numeric columns
    for header in [POPULARITY_COLUMN,REVENUE_COLUMN, RUNTIME_COLUMN,
                   VOTE_AVERAGE_COLUMN, VOTE_COUNT_COLUMN, TIMESTAMP_COLUMN]:
        feature_columns.append(tf.feature_column.numeric_column(header))
        feature_layer_inputs[header]=tf.keras.Input(shape=(1,), name=header)

    print("User Items count:")
    NUM_ITEMS = np.int(np.ceil(num_item_ids/np.sqrt(_ITEM_EMBEDDING_DIM))*np.sqrt(_ITEM_EMBEDDING_DIM))
    print(NUM_ITEMS)
    NUM_USERS = np.int(np.ceil(num_user_ids/np.sqrt(_USER_EMBEDDING_DIM))*np.sqrt(_USER_EMBEDDING_DIM))
    print(NUM_USERS)

    user_id = tf.feature_column.categorical_column_with_vocabulary_list(
        USER_COLUMN, range(1, NUM_USERS))
    user_embedding = tf.feature_column.embedding_column(
        user_id, _USER_EMBEDDING_DIM, max_norm=np.sqrt(_USER_EMBEDDING_DIM))

    feature_columns.append(user_embedding)
    feature_layer_inputs[USER_COLUMN]=tf.keras.Input(shape=(1,), name=USER_COLUMN, dtype=tf.int64)

    item_id = tf.feature_column.categorical_column_with_vocabulary_list(
        ITEM_COLUMN, range(1, NUM_ITEMS))
    item_embedding = tf.feature_column.embedding_column(
        item_id, _ITEM_EMBEDDING_DIM, max_norm=np.sqrt(_ITEM_EMBEDDING_DIM))

    feature_columns.append(item_embedding)
    feature_layer_inputs[ITEM_COLUMN] = tf.keras.Input(shape=(1,), name=ITEM_COLUMN, dtype=tf.int64)

    genres = tf.feature_column.numeric_column(
        GENRE_COLUMN, shape=(N_GENRE,), dtype=tf.uint8)
    feature_columns.append(genres)
    feature_layer_inputs[GENRE_COLUMN] = tf.keras.Input(shape=(N_GENRE,), name=GENRE_COLUMN)

    overviews = tf.feature_column.numeric_column(OVERVIEW_COLUMN, shape=(NUMBER_TEXT_FEATURES,), dtype=tf.float64)
    feature_columns.append(overviews)
    feature_layer_inputs[OVERVIEW_COLUMN] = tf.keras.Input(shape=(NUMBER_TEXT_FEATURES,), name=OVERVIEW_COLUMN)

    # popularity = tf.feature_column.numeric_column(POPULARITY_COLUMN)
    #
    # revenue = tf.feature_column.numeric_column(REVENUE_COLUMN)
    #
    # runtime = tf.feature_column.numeric_column(RUNTIME_COLUMN)
    #
    # vote_average = tf.feature_column.numeric_column(VOTE_AVERAGE_COLUMN)
    #
    # vote_count = tf.feature_column.numeric_column(VOTE_COUNT_COLUMN)

    # feature_columns = [user_embedding, item_embedding, time,
    #                    # genres,overviews,
    #                    popularity, revenue, runtime, vote_average, vote_count]


    return feature_columns, feature_layer_inputs

def _deserialize(examples_serialized):
  features = tf.io.parse_example(examples_serialized, _FEATURE_MAP)
  if MAX_RATING==5:
    feature_ratings = features[RATING_COLUMN] / MAX_RATING
  else:
      feature_ratings = tf.dtypes.cast(((features[RATING_COLUMN] / MAX_RATING) - 1), tf.int32)

  return features, feature_ratings

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

def construct_input_fns(batch_size=16, repeat=1, sample_size=100000, validation=False):

    original_dataset = construct_df(sample_size)

    new_dataset, user_map, movie_map = rebaseline_dataset(original_dataset)

    num_user_ids = len(user_map)
    num_item_ids = len(movie_map)

    if(validation==True):
        train_eval_df = new_dataset.sample(frac=0.9, random_state=0)
        train_df = train_eval_df.sample(frac=0.95, random_state=0)
        eval_df = train_eval_df.drop(train_df.index)
        predict_df = new_dataset.drop(train_eval_df.index)
        predict_df = predict_df.reset_index(drop=True)

        train_input_fn = df_to_input_fn(df=train_df, name="train", batch_size=batch_size, repeat=repeat,
                                        shuffle=sample_size)

        eval_input_fn = df_to_input_fn(df=eval_df, name="eval", batch_size=batch_size, repeat=repeat, shuffle=None)

        predict_input_fn = df_to_input_fn(df=predict_df, name="predict", batch_size=batch_size, repeat=repeat, shuffle=None)

        model_column_fn = functools.partial(construct_features, num_user_ids, num_item_ids)

        return train_input_fn, eval_input_fn, predict_input_fn, model_column_fn, user_map, movie_map

    else:
        train_df = new_dataset.sample(frac=0.8, random_state=0)
        eval_df = new_dataset.drop(train_df.index)
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

        train_input_fn = df_to_input_fn(df=train_df, name="train", batch_size=batch_size, repeat=repeat,shuffle=sample_size)

        eval_input_fn = df_to_input_fn(df=eval_df, name="eval", batch_size=batch_size, repeat=repeat, shuffle=None)

        model_column_fn = functools.partial(construct_features, num_user_ids, num_item_ids)

        return train_input_fn, eval_input_fn, model_column_fn, user_map, movie_map

def main():

    sample_size = 1000000
    dataset_name = 'c-1m'

    DATASET_PATH = os.path.join(ROOT_DIR, 'data','custom_datasets', dataset_name)

    train_df, test_df, val_df, predict_df, user_map, movie_map = \
        construct_movielens_df(val_split=.01, test_split=.05, sample_size=sample_size)

    train_df.to_csv(os.path.join(DATASET_PATH, 'train_df.csv'), index=False, index_label=False)

    test_df.to_csv(os.path.join(DATASET_PATH, 'test_df.csv'), index=False, index_label=False)

    val_df.to_csv(os.path.join(DATASET_PATH, 'val_df.csv'), index=False, index_label=False)

    predict_df.to_csv(os.path.join(DATASET_PATH, 'predict_df.csv'), index=False, index_label=False)

    pd.DataFrame.from_dict(user_map, orient='index', columns=[USER_COLUMN, 'NEW_'+USER_COLUMN])\
        .to_csv(os.path.join(DATASET_PATH, 'usermap.csv'))

    pd.DataFrame.from_dict(movie_map, orient='index', columns=[ITEM_COLUMN, 'NEW_'+ITEM_COLUMN])\
        .to_csv(os.path.join(DATASET_PATH, 'moviemap.csv'))


main()





