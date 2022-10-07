##########################################
#This is my implementation of a Hybrid version of NeuMF. The paper presenting NeuMF 
#is avaliable here: 
#I have included a model I trained on the NCC in my submission as it takes quite a while to train
##########################################
import numpy as np
import pandas as pd 
import keras
from keras import backend as K
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout, concatenate, Multiply
from tensorflow.keras.optimizers import Adam


movie_vectors = pd.read_csv('movie_vectors.csv')
user_vectors = pd.read_csv('users_vectors.csv')
ratings_df = pd.read_csv('ratings.csv')
user_df = pd.read_csv('users.csv')
movies_df = pd.read_csv('movies.csv')

movie_vectors = movie_vectors.drop(columns = 'Unnamed: 0')
user_vectors = user_vectors.drop(columns = 'Unnamed: 0')



num_users = 6040
num_movies = 3952


def there_exisits_an_entry(user, movie):

  temp_df = ratings_df[(ratings_df['user_id'] == user) & (ratings_df['movie_id'] == movie)]

  if len(temp_df) == 0:
    return False

  return True

def return_training_data():
    user_input, item_input, labels = [],[],[]
    for r in range(len(ratings_df)):
        row = ratings_df.iloc[r]
        u = row['user_id']
        i = row['movie_id']

        #Positive Instances
        try:
          #Append the user and movie vectors
          user_list = list(user_vectors[user_vectors['identifier'] == u].iloc[0])
          item_list = list(movie_vectors[movie_vectors['identifier'] == i].iloc[0])
          user_input.append(user_list)
          item_input.append(item_list)
          labels.append(1)
        except IndexError:
          pass
    
        #Negative Instances
        for t in range(4):
            h = np.random.randint(ratings_df.shape[1])
            while there_exisits_an_entry(u, h):
                h = np.random.randint(ratings_df.shape[1])
            try:
              #Append user and movie vectors
              user_list = list(user_vectors[user_vectors['identifier'] == u].iloc[0])
              item_list = list(movie_vectors[movie_vectors['identifier'] == i].iloc[0])
              user_input.append(user_list)
              item_input.append(item_list)
              labels.append(0)
            except IndexError:
              pass

        
    return user_input, item_input, labels






def create_model(num_users, num_items):

    user_vector, movie_vector = Input(shape=(8,), dtype='int32'), Input(shape=(8,), dtype='int32')
    GMF_user = Embedding(input_dim = num_users + 1, output_dim = 10, embeddings_initializer='normal', embeddings_regularizer = l2(0), input_length=10)
    GMF_movie = Embedding(input_dim = num_items + 1, output_dim = 10,  embeddings_initializer='normal', embeddings_regularizer = l2(0), input_length=10)
    MLP_user = Embedding(input_dim = num_users + 1, output_dim = 5,  embeddings_initializer='normal', embeddings_regularizer = l2(0), input_length=10)
    MLP_movie = Embedding(input_dim = num_items+ 1, output_dim = 5, embeddings_initializer='normal', embeddings_regularizer = l2(0), input_length=10)
    mf_item_latent = Flatten()(GMF_movie(movie_vector))
    mf_user_latent = Flatten()(GMF_user(user_vector))
    
    keras.backend.cast(mf_item_latent, dtype= 'float32')
    keras.backend.cast(mf_user_latent, dtype= 'float32' )
    
    mf_vector = Multiply()([mf_user_latent, mf_item_latent]) 
    latent_user_from_MLP = Flatten()(MLP_user(user_vector))
    Latent_movie_vector_from_MLP = Flatten()(MLP_movie(movie_vector))
    mlp_vector = concatenate([latent_user_from_MLP, Latent_movie_vector_from_MLP])
    
    layer = Dense(10, kernel_regularizer= l2(0), activation='relu')
    mlp_vector = layer(mlp_vector)

    predict_vector = concatenate([mf_vector, mlp_vector])
    predict = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    model = Model(inputs=[user_vector, movie_vector], outputs=predict)
    
    return model


model = create_model(num_users, num_movies)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', run_eagerly=True )
user_train, movie_train, labels_train = return_training_data()
user_vectors = np.array(user_train).reshape(-1, 8)
movie_vectors = np.array(movie_train).reshape(-1, 8)
labels = np.array(labels_train)
model.fit([user_vectors, movie_vectors], labels, batch_size=256, epochs=50, verbose=1, shuffle=True)


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