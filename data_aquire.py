import pandas as pd
import os
import requests
import json
import numpy as np
from time import sleep
import io
import zipfile
from sklearn.feature_extraction.text import TfidfTransformer
from datetime import datetime

##################################################################################
#### Function retreives data from stored csv files from movielens datasets
##################################################################################
def download_data_file(dataset):

    if dataset=='100k':
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    elif dataset=='25m':
        url = 'http://files.grouplens.org/datasets/movielens/ml-25m.zip'
    else:
        print("Not a valid dataset selected.")

    r = requests.get(url)

    z = zipfile.ZipFile(io.BytesIO(r.content))

    z.extractall(path=r'data')


def get_data_file(filename, dataset, path, print_head=True):

    if not os.path.exists(path):
        download_data_file(dataset)

    file_headers=pd.read_csv(r'data\file_headers.csv', sep=';',index_col=False)

    file_headers = file_headers.loc[file_headers['filename'] == filename]


    headings = list(str(file_headers.iloc[0,3]).split(','))

    sep = file_headers.iloc[0,2]

    df = pd.read_csv(os.path.join(path,filename), sep=sep, names=headings, encoding='latin-1')

    if print_head:
        print(df.head())

    return df


def get_100k_data():

    users = get_data_file(filename='u.user', dataset='100k', path=r'data\ml-100k', print_head=True)
    ratings = get_data_file(filename='u.data', dataset='100k', path=r'data\ml-100k', print_head=True)
    movies = get_data_file(filename='u.item', dataset='100k', path=r'data\ml-100k', print_head=True)

    return users, ratings, movies

def create_100k_embeddings():

    users, ratings, movies = get_100k_data()

    #cols_to_use = ratings.columns.difference(movies.columns)
    user_movie_embedding = pd.merge(ratings, movies, left_on='item_id', right_on='movie_id',
                                    how='left', suffixes=['_ratings', '_movies'])

    #cols_to_use = user_movie_embedding.columns.difference(users.columns)
    user_movie_embedding = pd.merge(user_movie_embedding, users, left_on='user_id', right_on='user_id',
                                    how='left', suffixes=['_ratings', '_users'])

    #user_movie_embedding['datetime'] = datetime.utcfromtimestamp(int(user_movie_embedding['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
    user_movie_embedding['datetime'] = pd.to_datetime(user_movie_embedding['timestamp'], unit='s')

    user_movie_embedding['days_after_release_rating']=(
            pd.to_datetime(user_movie_embedding['datetime'])-
            pd.to_datetime(user_movie_embedding['release_date'])
    ).dt.days

    keep_columns = ['user_id',
                    'item_id',
                    'rating',
                    'days_after_release_rating',
                    'unknown',
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
                    'Musical',
                    'Mystery',
                    'Romance',
                    'Sci-Fi',
                    'Thriller',
                    'War',
                    'Western',
                    'age',
                    'gender',
                    'occupation',
                    'zip_code']

    user_movie_embedding = user_movie_embedding.loc[:, [x in keep_columns for x in user_movie_embedding.columns]]

    # print(user_movie_embedding)

    user_movie_embedding = pd.get_dummies(user_movie_embedding)

    # print(user_movie_embedding)

    return user_movie_embedding


def get_25M_data(path=r'data\ml-25m', print_head=True):

    directory= path

        #r'C:\Users\brent\Documents\Projects\fastai\my-movie-recommender\data\ml-25m'

    # read in 1. genome_scores
    path_data = os.path.join(directory, 'genome-scores.csv')
    genome_scores = pd.read_csv(path_data, sep=",", header=0)
    if print_head:
        print("genome_scores:")
        print(genome_scores.head())

    # read in 2. genome_tags
    path_data = os.path.join(directory, 'genome-tags.csv')
    genome_tags = pd.read_csv(path_data, sep=",", header=0)

    if print_head:
        print("genome_tags:")
        print(genome_tags.head())

    # read in 3. links
    path_data = os.path.join(directory, 'links.csv')
    links = pd.read_csv(path_data, sep=",", header=0)

    if print_head:
        print("links:")
        print(links.head())

    # read in 4. movies
    path_data = os.path.join(directory, 'movies.csv')
    movies = pd.read_csv(path_data, sep=",", header=0)

    if print_head:
        print("movies:")
        print(movies.head())
        print(movies.columns)

    # read in 5. ratings
    path_data = os.path.join(directory, 'ratings.csv')
    ratings = pd.read_csv(path_data, sep=",", header=0)

    if print_head:
        print("ratings:")
        print(ratings.head())

    # read in 6. tags
    path_data = os.path.join(directory, 'tags.csv')
    tags = pd.read_csv(path_data, sep=",", header=0)

    if print_head:
        print("tags:")
        print(tags.head())

    # read in 7. overview
    imdb_directory=r'data\imdb'
    path_data = os.path.join(directory, 'movie_overview.csv')
    overviews = pd.read_csv(path_data, sep=",", header=0)

    if print_head:
        print("overviews:")
        print(overviews.head())

    return genome_scores, genome_tags, links, movies, ratings, tags, overviews

def get_tmdb_overview(tmdb_id):

    keys = pd.read_csv(r'auth_keys\api_keys.csv', sep=',', index_col=False)

    api_key = keys.loc[0]['key']

    query_url = 'https://api.themoviedb.org/3/movie/'+ \
                str(tmdb_id) + \
                '?api_key=' + \
                str(api_key)

    r = requests.get(query_url)

    tmdb_json = r.json()

    try:
        ret_overview = tmdb_json['overview']
    except:
        ret_overview='No overview found.'
    finally:
        return ret_overview


def set_env_key_value(key, value):

    os.environ[key]=value

def aquire_overviews(links):
    tmbd_overview = pd.DataFrame(columns=['movieId', 'imdbId', 'tmbdId', 'overview'])

    j = 0
    count = len(links['tmdbId'])
    print("Total Count:" + str(count))

    for i in range(count):

        link = links.iloc[i]
        j += 1
        # movie_id = -1 if np.nan(int(link['movieId'])) else int(link['movieId'])
        # imdb_id = -1 if np.nan(int(link['imdbId'])) else int(link['imdbId'])
        # tmdb_id = -1 if np.nan(int(link['tmdbId'])) else int(link['tmdbId'])

        movie_id = str(link['movieId'])
        imdb_id = str(link['imdbId'])
        tmdb_id = str(link['tmdbId'])

        overview = get_tmdb_overview(tmdb_id)

        cur_df = pd.DataFrame(data=[[movie_id, imdb_id, tmdb_id, overview]],
                              columns=['movieId', 'imdbId', 'tmbdId', 'overview'])
        tmbd_overview = tmbd_overview.append(cur_df)

        randomMilli = np.random.randint(low=25, high=50)
        sleep(float(randomMilli) / 1000.)
        if j == 100:
            j = 0
            print(str(i + 1) + str(":") + str(round(float(i) / float(count) * 100., 2)) + str("%"))

    tmbd_overview.to_csv('data/imdb/movie_overview.csv', index=False, index_label=False)


def main():
    # genome_scores, genome_tags, links, movies, ratings, tags, overviews = get_data(
    #                     path=r'C:\Users\brent\Documents\Projects\fastai\my-movie-recommender\data\ml-25m',
    #                     print_head=False
    # )
    user_movie_embedding = create_100k_embeddings()

    print(user_movie_embedding)



main()