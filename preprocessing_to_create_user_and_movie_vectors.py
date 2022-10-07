###########################
# This is the python file I used to create the latent vectors for the users and movies for the NeuMF Hybrid 
# It takes quite a while to run (I ran it on the NCC) and requires some finicky packages so I have included the results 
#In the user and movie vector csv files 


###########################


#import pgeocode
import pandas as pd 
from bs4 import BeautifulSoup 
import requests 
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imdb import IMDb
from sentence_transformers import SentenceTransformer
from skimage import io
import cv2
import tensorflow as tf
import keras
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
from skimage.transform import resize





df_movies = pd.read_csv("movies.csv")
df_ratings = pd.read_csv("ratings.csv")
df_users = pd.read_csv("users.csv")



#BERT MODEL 
def return_description(url):
    print(url)
    if url == None:
         return None 
    
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "lxml")
    subsoup = soup.find("span", {'class':'GenresAndPlot__TextContainerBreakpointL-sc-cum89p-1 eqlIrG'})
    
    if subsoup == None:
        return None 
    
    for sub in subsoup:
        print(sub)
        try:
          return str(sub)
        except:
          return None
    

sbert_model = SentenceTransformer('all-mpnet-base-v2')
def sentence_encoding(url):
    sentence = return_description(url)
    if sentence == None:
      sentence_vector =[0] * 768
  
    else:
      sentence_vector = sbert_model.encode(sentence)
    
    return list(sentence_vector)
 


movie_df = pd.read_csv('movies.csv')
movie_url = pd.read_csv('movie_url.csv', names = ['Identifier', 'Url'])



lst = []   
for i in range(len(movie_url)):
    url = movie_url.iloc[i]['Url']    
    identifier = movie_url.iloc[i]['Identifier']  
    list_encoding = sentence_encoding(url)
    list_encoding.insert(0, identifier)
    lst.append(list_encoding)
    
sentence_encoding = pd.DataFrame(lst)
sentence_encoding.to_csv('encoding.csv')

#movie_url['description'] = movie_url['Url'].apply(return_description)
#movie_url.to_csv('movie_processed.csv')
#movie_url['director'] = movie_url['Url'].apply(return_director)
#movie_url.to_csv('/content/drive/MyDrive/movies/movie_processed.csv')
#movie_url['star'] = movie_url['Url'].apply(return_star)
#movie_url.to_csv('/content/drive/MyDrive/movies/movie_processed.csv')


def load_image(amazon_url, resized_fac = 0.1):
    image = io.imread(amazon_url)
    w, h, _ = image.shape
    resized = cv2.resize(image, (int(h*resized_fac), int(w*resized_fac)), interpolation = cv2.INTER_AREA)
    return resized




from PIL import Image
movie_posters = pd.read_csv('movie_poster.csv', names = ['Identifier', 'Url'])


base_model = ResNet50(weights='imagenet', include_top=False, input_shape = (300,200, 3))
base_model.trainable = False

# Add Layer Embedding
model = keras.Sequential([base_model,GlobalMaxPooling2D()])


def get_embedding(model, amazon_link, resized_fac = 0.1):

    try:
      img = io.imread(amazon_link)
    except:
      return [0] * 2048
    
    x = resize(img, (300, 200),anti_aliasing=True)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    try:
      prediction = model.predict(x).reshape(-1)

    except:
      return [0] * 2048
    
    return list(prediction)

lst = [] 

for i in range(len(movie_posters)):
    url = movie_posters.iloc[i]['Url']
    identifier = movie_posters.iloc[i]['Identifier']
    embedding = get_embedding(model, url)
    embedding.insert(0, identifier)
    lst.append(embedding)
    
df = pd.DataFrame(lst)
df.to_csv('image_embedding.csv')



#Reduce dimensionality with autoencoder 
image_embedding = pd.read_csv('image_embedding.csv')
sentence_embedding = pd.read_csv('encoding.csv')


image_embedding = image_embedding.fillna(value=0)
sentence_embedding = sentence_embedding.fillna(value=0)

image_embedding = image_embedding.drop(columns = ['Unnamed: 0'])
sentence_embedding = sentence_embedding.drop(columns = ['Unnamed: 0'])


image_embedding.set_index('0', inplace = True)
sentence_embedding.set_index('0', inplace = True)

combined_df = pd.concat([image_embedding, sentence_embedding], axis=1)


combined_df = pd.read_csv('processed_users.csv')
combined_df.set_index('user_id')
x_train = combined_df.sample(frac=0.8, random_state=200) #random state is a seed value
x_test = combined_df.drop(x_train.index)




def scale_datasets(x_train, x_test):

  standard_scaler = MinMaxScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )
  return x_train_scaled, x_test_scaled


class AutoEncoders(Model):

  def __init__(self, output_units):

    super().__init__()
    self.encoder = Sequential(
        [
          Dense(32, activation="relu"),
          Dense(16, activation="relu"),
          Dense(7, activation="relu")
        ], name="sequential"
    )

    self.decoder = Sequential(
        [
          Dense(16, activation="relu"),
          Dense(32, activation="relu"),
          Dense(output_units, activation="sigmoid")
        ]
    )

  def call(self, inputs):
    
      encoded = self.encoder(inputs)
      decoded = self.decoder(encoded)
      return decoded



x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)
x_train_scaled = x_train_scaled.fillna(value = 0)
x_test_scaled = x_test_scaled.fillna(value =0 )

auto_encoder = AutoEncoders(len(x_train_scaled.columns))

auto_encoder.compile(
    loss='mae',
    metrics=['mae'],
    optimizer='adam'
)

history = auto_encoder.fit(x_train_scaled, x_train_scaled, epochs=15, batch_size=32, validation_data=(x_test_scaled, x_test_scaled))

standard_scaler = MinMaxScaler()

movie_scaled = pd.DataFrame(standard_scaler.fit_transform(combined_df),columns=combined_df.columns)

encoder_layer = auto_encoder.get_layer('sequential')
reduced_df = pd.DataFrame(encoder_layer.predict(movie_scaled))
reduced_df = reduced_df.add_prefix('feature_')
reduced_df['identifier'] = combined_df.index


reduced_df.to_csv('latent_users.csv')

"""

def convert_zip_lat(zipcode):

    nomi = pgeocode.Nominatim('us')
    lat = nomi.query_postal_code(str(zipcode))["latitude"]

    return lat

def convert_zip_lon(zipcode):

    nomi = pgeocode.Nominatim('us')
    lon = nomi.query_postal_code(str(zipcode))["longitude"]

    return lon

lat_column = df_users.apply(lambda row: convert_zip_lat(row.zip_code), axis=1)
lon_column = df_users.apply(lambda row: convert_zip_lon(row.zip_code), axis=1)

occupation_dict = {0:  "other", 1:  "academic/educator",
                2:  "artist", 3:  "clerical/admin",
                4:  "college/grad student", 5:  "customer service",
                6:  "doctor/health care", 7:  "executive/managerial",
                8:  "farmer", 9:  "homemaker",
                10:  "K-12 student", 11:  "lawyer",
                12:  "programmer", 13:  "retired",
                14:  "sales/marketing", 15:  "scientist",
                16:  "self-employed", 17:  "technician/engineer",
                18:  "tradesman/craftsman", 19:  "unemployed",
                20:  "writer" }


df_users["latitude"] = lat_column
df_users["longitude"] = lon_column
df_users.replace({"occupation": occupation_dict}, inplace=True)
df_users = pd.get_dummies(df_users, columns = ['gender', 'occupation'])
df_users = df_users.drop(columns = ['zip_code', 'Unnamed: 0'])
df_users.to_csv('/content/drive/MyDrive/movies/processed_users.csv')



# create an instance of the IMDb class
ia = IMDb()

def return_IMDB_identifier(url):
    
    split = url.split('/')[-2]
    identifier = split.replace('tt', '')
    
    return identifier 


def return_director(imdb_url):
    try:
        identifier = return_IMDB_identifier(imdb_url)
        movie = ia.get_movie(identifier)
    except: 
        
        return None 
    
    try:
        for director in movie['directors']:
            return director['name']
    except:
        return None
        
def return_star(imdb_url):     

    try: 
      identifier = return_IMDB_identifier(imdb_url)
      movie = ia.get_movie(identifier)
      
      cast = movie.get('cast')
    except:
      return None
    
    try:
        for actor in cast:
            return actor['name']
    
    except:
        return None
"""