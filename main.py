from classesfor import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
X >> matrix of movies, there are 4778 movies and each movie has 10 features 
W >> matrix of parameters, for 443 users there are 10 feature for each
b >> matrix of biases, there are totally 443, a bias for a person
R >> matrix of is_rated, whether a movie has been rated from user or not(0 or 1)
Y >> matrix of rates, rates from (0-5)
"""

X, W, b, num_movies, num_features, num_users = load_specs()
Y, R = load_ratings()


# lambda as regularization, x w b y as movies parameters bias and rates
def cost_func(movies, parameters, bias, rates, is_rated, _lambda):
    j = (tf.linalg.matmul(movies, tf.transpose(parameters)) + bias - rates) * is_rated
    j = 0.5 * tf.reduce_sum(j ** 2) + (_lambda / 2) * (tf.reduce_sum(movies ** 2) + tf.reduce_sum(parameters ** 2))
    return j


J = cost_func(X, W, b, Y, R, 0)
print(J)

"""
assume that I am a new user and rate some movies of the list,
then it should predict the rates of the other movies in the list base on my behavior.
"""
movieList, movieList_df = load_movie_list_pd()
# initialize my ratings
my_ratings = np.zeros(num_movies)
my_ratings[2700] = 5  # for example id 2700 in movies_X represent Toy Story 3, and we rate it '5'
my_ratings[2609] = 2  # Persuasion (2007)
my_ratings[929] = 5  # Lord of the Rings: The Return of the King, The
my_ratings[246] = 5  # Shrek (2001)
my_ratings[2716] = 3  # Inception
my_ratings[1150] = 5  # Incredible, The (2004)
my_ratings[382] = 2  # Amelie (Fabulous destin d'Amélie Poul ain, Le)
my_ratings[366] = 5  # Harry Potter and the Sorcerer's Stone (2001)
my_ratings[622] = 5  # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988] = 3  # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1  # Louis The roux: Law & Disorder (2008)
my_ratings[2937] = 1  # Nothing to Declare (Rien à déclarer)
my_ratings[793] = 5  # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]
# lest add these to Y and R and normalize it
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0).astype(int), R]
y_norm, y_mean = normalize_ratings(Y, R)
num_features = 100
num_movies, num_users = Y.shape
# set initial parameters (W, X), use tf.Variable to track these variables, and set the optimizer
tf.random.set_seed(1234)  # for consistent results
W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 1000  # I choose it due to prevent vanishing gradient
lambda_ = 1
# training our model
for iter_ in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cost_func(X, W, b, y_norm, R, lambda_)

    grads = tape.gradient(cost_value, [X, W, b])

    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if iter_ % 20 == 0:
        print(f"Training loss at iteration {iter_}: {cost_value:0.1f}")

"""
recommendation, now we trained parameters(weights and biases) so that we could make prediction best on my ratings
humour
"""
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
pm = p + y_mean
my_predictions = pm[:, 0]
ix = tf.argsort(my_predictions, direction='DESCENDING')
for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')
filter_ = (movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
movieList_df.loc[ix[:300]].loc[filter_].sort_values("mean rating", ascending=False)
