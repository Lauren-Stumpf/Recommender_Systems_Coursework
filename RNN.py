#############################################################
# RECURRENT RECOMMENDER NETWORKS
#This is my implementation of the paper Recurrent Recommender Networks as described in https://research.google/pubs/pub45881/
#I have included the model I trained on the NCC as it takes a while to train
#############################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam



user_df = pd.read_csv('users.csv')
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
num_users = 6040
num_movies = 3952

ratings_df = ratings_df.drop(columns = ['Unnamed: 0'])

user_df = user_df.drop(columns = ['zip_code', 'Unnamed: 0'])
user_df['gender'] = user_df['gender'].replace({'M': 0, 'F': 1})
user_df['age'] = user_df['age'].replace({1: 0, 35: 1, 45: 2, 50: 3, 18: 4, 56: 5, 25: 6})

movies_df = movies_df.drop(columns = ['title', 'Unnamed: 0'])
movies_dictionary = {'Comedy': 0, 'Horror': 1, "Children's": 2, 'Romance': 3, 'Crime': 4, 'Mystery': 5, 'Film-Noir': 6, 'Drama': 7, 'Adventure': 8, 'Fantasy': 9, 'War': 10, 'Western': 11, 'Documentary': 12, 'Action': 13, 'Sci-Fi': 14, 'Animation': 15, 'Musical': 16, 'Thriller': 17}
movies_map_genres_to_list_of_numbers = {genres: [movies_dictionary[row] for row in genres.split('|')] for num, genres in enumerate(set(movies_df['genres']))}
movies_df['genres'] = movies_df['genres'].map(movies_map_genres_to_list_of_numbers)


def split_dataset_in_time_order(sorted_df):
  num_of_samples = len(sorted_df)
  num_of_train_samples = int(num_of_samples * 0.7)
  num_of_test_samples = int(num_of_samples * 0.3)

  train = sorted_df.head(num_of_train_samples)
  test = sorted_df.tail(num_of_test_samples)

  return train, test


overall_df = pd.merge(pd.merge(ratings_df, user_df), movies_df)
overall_df = overall_df.sort_values(by=['timestamp'])
train, test = split_dataset_in_time_order(overall_df)

def create_model(num_users, num_items):
 
        n_step = 1

        userid = Input(shape=(1,), dtype='int32', name = 'userid')
        movieid = Input(shape=(1,), dtype='int32', name="movieid")

        #User embedding
        uid_onehot = tf.reshape(tf.one_hot(userid, 6040), shape=[-1, 6040])
        uid_layer = Dense(units=128, activation=tf.nn.relu)(uid_onehot)

        mid_onehot = tf.reshape(tf.one_hot(movieid, 3952), shape=[-1, 3952])
        mid_layer = Dense(units=128, activation=tf.nn.relu)(mid_onehot)
 
        mf_item_latent = tf.reshape(mid_layer , shape=[-1, n_step, 128])
        mf_user_latent = tf.reshape(uid_layer, shape=[-1, n_step, 128])

        userInput = tf.transpose(mf_item_latent, [1, 0, 2])
        movieInput = tf.transpose(mf_user_latent, [1, 0, 2])

        
        userCell = tf.keras.layers.GRUCell(units = 128)
        movieCell =  tf.keras.layers.GRUCell(units = 128)
        
        #RNN layer
        rnn_layer_user = tf.keras.layers.RNN(userCell, return_sequences=True, return_state=True)
        userOutputs, userStates = rnn_layer_user(userInput)
        userOutput = userOutputs[-1]

        rnn_layer_movie = tf.keras.layers.RNN(movieCell, return_sequences=True, return_state=True)
        movieOutputs, movieStates = rnn_layer_movie(movieInput)
        movieOutput = movieOutputs[-1]

        W_user = tf.keras.backend.random_normal(shape=[128, 64], stddev=0.1)
        W_movie = tf.keras.backend.random_normal(shape=[128, 64], stddev=0.1)

        b_user = tf.keras.backend.random_normal(shape=[64], stddev=0.1)
        b_movie = tf.keras.backend.random_normal(shape=[64], stddev=0.1)
        
        userVector = tf.matmul(userOutput, W_user)
        userVector = tf.add(userVector, b_user)

        movieVector = tf.matmul(movieOutput, W_movie)
        movieVector = tf.add(movieVector, b_movie)

        mf_vector = tf.multiply(userVector, movieVector) 

        pred = tf.reduce_sum(mf_vector, axis=1, keepdims=True)
        model = Model(inputs=[userid, movieid], outputs=pred)

        return model
    
    

model = create_model(num_users, num_movies)
num_epochs = 20
batch_size = 256
model.compile(optimizer=Adam(lr=0.01), loss=tf.keras.losses.MeanSquaredError())
user_input, item_input, labels = train['user_id'], train['movie_id'], train['rating']


model.fit([np.array(user_input), np.array(item_input)], np.array(labels), batch_size=batch_size, epochs=20, verbose=1)

#Precompute recommendations
list_of_users = user_df['user_id']
list_of_movies = movies_df['movie_id']

list_of_users_ = []
list_of_movies_ = []
for i in list_of_users:
  for j in list_of_movies: 
    list_of_users_.append(i)
    list_of_movies_.append(j)


num_users = max(list_of_users)
num_movies = max(list_of_movies)

ratings = model.predict([np.array(list_of_users_), np.array(list_of_movies_)])
num_movies = int(len(list_of_movies_) / num_users)
ratings = ratings.reshape(num_users, num_movies)

df = pd.DataFrame(ratings)
df.to_csv('user_item.csv')
