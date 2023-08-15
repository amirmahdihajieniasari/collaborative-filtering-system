import numpy as np
import pandas as pd
from numpy import loadtxt


def normalize_ratings(y, r):
    """
    preprocess data by subtracting mean rating for every movie (every row).
    only include ratings which R(i,j)=1.
    """
    y_mean = (np.sum(y * r, axis=1) / (np.sum(r, axis=1) + 1e-12)).reshape(-1, 1)
    y_norm = y - np.multiply(y_mean, r)
    return y_norm, y_mean


def test(*args):
    output = [loadtxt(ar, delimiter=",") for ar in args]
    return output


def load_specs():
    file = open('./datasets/movies_X.csv', 'rb')
    x = loadtxt(file, delimiter=",")

    file = open('./datasets/movies_W.csv', 'rb')
    w = loadtxt(file, delimiter=",")

    file = open('./datasets/movies_b.csv', 'rb')
    b = loadtxt(file, delimiter=",")
    b = b.reshape(1, -1)
    num_movies, num_features = x.shape
    num_users, _ = w.shape
    return x, w, b, num_movies, num_features, num_users


def load_ratings():
    file = open('./datasets/movies_Y.csv', 'rb')
    y = loadtxt(file, delimiter=",")

    file = open('./datasets/movies_R.csv', 'rb')
    r = loadtxt(file, delimiter=",")
    return y, r


def load_movie_list_pd():
    """ returns df with and index of movies in the order they are in the Y matrix """
    df = pd.read_csv('./datasets/movie_list.csv', header=0, index_col=0, delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return mlist, df
