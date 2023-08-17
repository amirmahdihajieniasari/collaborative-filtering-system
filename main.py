import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from classesfor import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""                                                                                                                                                                                                                                                                                                                                                              
item_train    >> data which represent movie id, year made, ave rating, is action, is adventure, ... of all 847 movies( 
 they maybe duplicate or even more cause movies rated time by users)

user_train    >> data which represent user id, rating count, ave rating, is action, ... of all users

y_rating      >> data which represent labeled y ratings

item_features >> data which represent features of movies(movie id, year, ave rating, is action, adventure, animation,
 children, comedy, crime, documentary, drama, fantasy, horror, mystery, romance, sci-fi, thriller)
 
user_features >> data which represent feature of users(user id, rating count, rating ave, action, adventure, animation,
 children, comedy, crime, documentary, drama, fantasy, horror, mystery, romance, sci-fi, thriller)
 
item_vecs     >> rather than item_train this data is not duplicate or more

user_vecs     >> data which represent our user with features of user_features but in scale of item_vecs 

movie_dict    >> data which represent dict of movie(key of title has value movie name and year and key of genre)

user_to_genre >> data which represent dict of keys&(values) per users:glist(av. of genres rating),
 g_count(sum of rates of genres in total), rating_count(movies that rated), rating_sum(sum of rates in total),
  movies(movies that rated with their value rating y), rating_ave(movies average rating)
"""

item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_spec()
num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating
num_item_features = item_train.shape[1] - 1  # remove movie id
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items

# preparing training data(scale training data)
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled = y_train
scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)
scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)
scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))

# split the data into training and test sets
item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test = train_test_split(y_train, train_size=0.80, shuffle=True, random_state=1)

# neural network
num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs)

])
item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs)

])
input_user = tf.keras.layers.Input(shape=num_user_features)
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)
input_item = tf.keras.layers.Input(shape=num_item_features)
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)
output = tf.keras.layers.Dot(axes=1)([vu, vm])  # compute the dot product of the two vectors vu and vm
model = tf.keras.Model([input_user, input_item], output)
model.summary()
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train,
          epochs=60)  # choose 60 for epoch to prevent vanishing gradiant
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

"""
prediction,
assume that I am a new user
"""
new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_rating_count = 3
user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])
# generate and replicate the user vector to match the number movies in the data set
user_vecs = gen_user_vecs(user_vec, len(item_vecs))
s_user_vecs = scalerUser.transform(user_vecs)
s_item_vecs = scalerItem.transform(item_vecs)
y_p = model.predict([s_user_vecs[:, u_s:], s_item_vecs[:, i_s:]])
# unscale y prediction
y_pu = scalerTarget.inverse_transform(y_p)
# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  # negate to get largest rating first
sorted_ypu = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  # using unscaled vectors for display
print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount=10)
