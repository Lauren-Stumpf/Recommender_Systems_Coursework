print('Starting up... ')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.optimizers import Adam
import numpy as np 
import pandas as pd

#Read in data and define functions
ratings_df = pd.read_csv('ratings.csv')
user_df = pd.read_csv('users.csv')
movies_df = pd.read_csv('movies.csv')

if 'Unnamed: 0' in ratings_df.columns:
    ratings_df = ratings_df.drop(columns = ['Unnamed: 0'])
    
if 'Unnamed: 0' in user_df.columns:
    user_df = user_df.drop(columns = ['Unnamed: 0'])
    
if 'Unnamed: 0' in movies_df .columns:
    movies_df = movies_df .drop(columns = ['Unnamed: 0'])
    

movie_url = pd.read_csv('movie_url.csv', names = ['movie_id', 'link']) 

def remove_whitespace_and_capitalise(string):
    string = string.strip()
    string = string.capitalize()
    
    return string


def check_if_valid_input(user_input, list_of_acceptable_options):
    string = user_input.strip()
    string = string.capitalize()
    
    if len(string) > 1:
        return False
    
    if string in list_of_acceptable_options:
        return True
            
    return False

def check_if_valid_user_id(user_id, num_users):
    try:
        user_id = int(user_id)
    except ValueError:
        return False
    
    if user_id <= 0 or user_id > num_users:
        return False
    
    return True

def check_if_valid_movie_id(movie_id, num_movies):
    try: 
        movie_id = int(movie_id)
        
    except ValueError:
        return False
    
    if movie_id <= 0 or movie_id > num_movies:
        return False
    
    return True 

def check_if_valid_rating(rating):
    try:
        rating = int(rating)
        
    except ValueError:
        return False
    
    if rating >= 1 and rating <= 5:
        return True
    
    return False

def check_if_valid_number(number):
    
    try:
        number = int(number)
        return True
    except:
        return False
    
def check_if_between_range(number, min_value, max_value):
    
    try: 
        number = int(number)
        if number > max_value or number < min_value:
            return False
        return True
    except:
        return False
    
def check_if_1_or_2(value):
    
    if int(value) == 1 or int(value) == 2:
        return True
    
    return False



print('########################################################################')
print('#                                                                      #')
print('#                    MovieLens Recommender System                      #')
print('#                                                                      #')
print('########################################################################')

new_user_flag = False
num_users = max(ratings_df['user_id'])
num_movies = max(ratings_df['movie_id'])
timestamp = max(ratings_df['timestamp'])

new_training_df = pd.DataFrame(columns = ['user_id', 'movie_id', 'rating', 'timestamp'])

print('Hello. Please choose a recommender system')
print('1 - Hybrid (Content-Collaborative)')
print('2 - Collaborative')
recommender_system = input()
accepted = check_if_1_or_2(recommender_system)
while accepted == False: 
    print('That is not a valid option. Please choose from the existing options')
    recommender_system = input()
    accepted = check_if_1_or_2(recommender_system)

print('Loading the model...')
if recommender_system == str(1):
    user_item_matrix = pd.read_csv('user_item_NeuMF.csv')
    user_item_matrix = user_item_matrix.drop(columns = ['Unnamed: 0'])
    user_item_matrix = user_item_matrix.T
    user_item_matrix.columns = range(1,6041)
    user_item_matrix.index = range(1, 3884)
    
    model = keras.models.load_model('model_NeuMF')
    
elif recommender_system == str(2):
    user_item_matrix = pd.read_csv('user_item_RNN.csv')
    user_item_matrix = user_item_matrix.drop(columns = ['Unnamed: 0'])
    user_item_matrix = user_item_matrix.T
    user_item_matrix.columns = range(1,6041)
    user_item_matrix.index = range(1, 3884)
    
    model = keras.models.load_model('model_RNN')



#temp_ratings_df = ratings_df[(ratings_df['movie_id'] == movie_id) & (ratings_df['rating'] == rating)]

print('Hello. Please identify yourself by selecting the appropriate option')
print('E - Existing User')
print('N - New User')
print('X - Exit')

x = input()

accepted = check_if_valid_input(x, ['E', 'N', 'X'])
while accepted == False:
        print('That is not a valid option. Please choose from the existing options')
        x = input()
        accepted = check_if_valid_input(x, ['E', 'N', 'X'])
x = remove_whitespace_and_capitalise(x)      
if x == 'X':
    quit()

elif x == 'E': 
    print('Please select the appropriate option')
    print('N - Input User ID')
    print('I - Identify User by inputting movies the User likes')
    print('X - Exit')
    
    x = input()
    accepted = check_if_valid_input(x, ['N', 'I', 'X'])
    while accepted == False:
        print('That is not a valid option. Please choose from the existing options')
        x = input()
        accepted = check_if_valid_input(x, ['N', 'I', 'X'])
    x = remove_whitespace_and_capitalise(x)
    
    if x == 'X': 
        quit()
    
    elif x == 'N': 
        
        print('Please choose a User ID from between 1 and ' + str(num_users) +':')
        x = input()
        
        accepted = check_if_valid_user_id(x, num_users)
        while accepted == False:
            print('Please enter a valid user ID')
            x = input()
            accepted = check_if_valid_user_id(x, num_users)
        x = remove_whitespace_and_capitalise(x)
         
        user_id = x
    
    elif x == 'I': 
        
        movie_id_list = []
        rating_list = []
        while True:
        
            print('Please input a Movie ID between 1 and ' + str(num_movies) +':')
            movie_id = input()
            
            accepted = check_if_valid_movie_id(movie_id, num_movies)
            while accepted == False:
                print('Please enter a valid movie ID')
                movie_id = input()
                accepted = check_if_valid_movie_id(movie_id, num_movies)
            movie_id = remove_whitespace_and_capitalise(movie_id)
             
            print('Please input a rating between 1 and 5:')
            rating = input()
            accepted = check_if_valid_rating(rating)
            
            while accepted == False:
                print('Please enter a valid rating')
                rating = input()
                accepted = check_if_valid_rating(rating)
            rating = remove_whitespace_and_capitalise(rating)
            
            movie_id_list.append(int(movie_id))
            rating_list.append(int(rating))
        
            
            print('Would you like to enter another movie')
            print('Y - yes')
            print('N - no')
            x = input() 
            
            accepted = check_if_valid_input(x, ['Y', 'N'])
            while accepted == False:
                print('Please enter a valid option')
                x = input()
                accepted = check_if_valid_input(x, ['Y', 'N'])
            x = remove_whitespace_and_capitalise(x)
                
            if x == 'N':
                break

        temp_ratings_df = ratings_df[ratings_df['movie_id'].isin(movie_id_list)]
        temp_ratings_df = temp_ratings_df[temp_ratings_df['rating'].isin(rating_list)]
            
        if len(temp_ratings_df) == 0:
            print('No matching Users.')
            quit()
        
        
        elif len(temp_ratings_df) < 5:
            num_of_options_to_display = len(temp_ratings_df)
            
        else: 
            num_of_options_to_display = 5
            
        list_of_temp_ratings = list(set(temp_ratings_df['user_id'].to_list()))
        print('There are ' + str(len(list_of_temp_ratings)) + ' users matching this description. Please input how many you would like to see')
        
        num_of_options_to_display = input()
        accepted = check_if_between_range(num_of_options_to_display, 0, len(temp_ratings_df))
        while accepted == False:
            print('Please enter a valid number')
            num_of_options_to_display = input()
            accepted = check_if_between_range(num_of_options_to_display, 0, len(temp_ratings_df))
        num_of_options_to_display = remove_whitespace_and_capitalise(num_of_options_to_display)
        num_of_options_to_display = int(num_of_options_to_display)
        
        print('List of users matching these items are:')
        print(list_of_temp_ratings[:num_of_options_to_display])
        
        print('Please choose a user ID')
        x = input()
        accepted = check_if_valid_user_id(x, num_users)
        while accepted == False:
            print('Please enter a valid user ID')
            x = input()
            accepted = check_if_valid_user_id(x, num_users)
        x = remove_whitespace_and_capitalise(x)
        
        user_id = x 
        
        

elif x == 'N' or x == 'n': 
    num_users += 1
    print('Created new user ' + str(num_users))
    user_id = num_users
    new_user_flag = True
    ratings_df.iloc[num_users] = [user_id,	None,	None,	timestamp  + 1]
    


print('Hello, User: ', user_id)
print('Please select the appropriate option')
print('D - How your data is collected and used')
print('R - Get recommendations')
print('U - Update User Profile')
print('X - Exit')

x = input()
accepted = check_if_valid_input(x, ['D', 'R', 'U', 'X'])
while accepted == False:
    print('That is not a valid option. Please choose from the existing options')
    x = input()
    accepted = check_if_valid_input(x, ['D', 'R', 'U', 'X'])
x = remove_whitespace_and_capitalise(x)   

if x == 'X':
    quit()

elif x == 'D':
    print('Your data is collected in the accordance with the Data Protection Act 2018. The data that is stored is the user id, the occupation of the user, the gender of user, the age of the user and the zip-code of the user. It transform these features and uses this to train one recommender system. Data regarding the rating the user gave a movie at a certain timestamp is also stored. All data is stored with the purpose of producing dependable and relevant recommendations for the user. The user can remove their data at any point by navigating to the update user profile section and delete entries')
    
elif x == 'R': 
    cumulative_recommendations = 0
    flag = True
    while flag:
        print('Please enter a number for how many recommendations you want:')
        num_of_rec = input()
        accepted = check_if_valid_number(num_of_rec)
        while accepted == False:
            print('Please enter a valid number')
            number_of_rec = input()
            accepted = check_if_valid_number(num_of_rec)
        num_of_rec = int(num_of_rec)

        try: 
            user_ratings = user_item_matrix.sort_values(by=[int(user_id)], ascending = False)[int(user_id)]
            movie_list = list(user_ratings.index)
            user_ratings = list(user_ratings.values)
         
        #new user
        except KeyError:
            model = keras.models.load_model('model_RNN')
            list_of_movies = np.array(movies_df['movie_id'])
            list_of_user_id = np.array([user_id] * len(list_of_movies))
            
            predictions = model.predict([list_of_user_id, list_of_movies])

            predictions = predictions.tolist()
            predictions = [item for sublist in predictions for item in sublist]
            user_ratings = sorted(predictions, reverse=True)

            movie_list = [x for _, x in sorted(zip(predictions, list_of_movies))]

        
        if len(movie_list) < num_of_rec:
            num_of_rec = len(movie_list) - 1
        
        
        movie_list = movie_list[cumulative_recommendations:num_of_rec + cumulative_recommendations]
        ratings = user_ratings[cumulative_recommendations:num_of_rec + cumulative_recommendations]
        movie_titles = []
        for movie_id in movie_list:
            row = movies_df[movies_df['movie_id'] == movie_id]['title'].values
            movie_titles.append(row[0])
        
        
        for i in range(num_of_rec):
            movie = movie_titles[i]
            rating = round(ratings[i] / 5, 2) * 100
            print('Recommendation ' + str(i + cumulative_recommendations + 1) + '. ' + movie + ' with a confidence of ' + str(rating) + '%')
        

        df_temp = ratings_df[ratings_df['user_id'] == int(user_id)]
        df_temp = df_temp.sort_values(by=['rating'], ascending=False)
        df_temp = df_temp.head()
        list_of_movies = list(df_temp['movie_id'])
        cumulative_recommendations += num_of_rec
        list_of_movie_names = []
        for movie_id in list_of_movies:
            row = movies_df[movies_df['movie_id'] == movie_id]['title'].values
            list_of_movie_names.append(row)
        
        try:
            print('\n')
            print('These recommendations have been generated because you rated the films '  + ", ".join(str(movie[0]) for movie in list_of_movie_names) +' highly ')
            print('\n')
        except IndexError:
            pass
        print('Please select the appropriate option')
        print('M - More recommendations')
        print('I - Information on the movies presented')
        print('X - Exit')
        
        x = input()
        accepted = check_if_valid_input(x, ['M', 'I', 'X'])
        while accepted == False:
            print('That is not a valid option. Please choose from the existing options')
            x = input()
            accepted = check_if_valid_input(x, ['R', 'C', 'X'])
        x = remove_whitespace_and_capitalise(x)
        if x == 'X':
            quit()
        
        elif x == 'I':
            print('Please enter the movie:')
            movie_ = input()

            row = movies_df[movies_df['title'] == movie_]

            
            while len(row) == 0: 
                print('Sorry that movie is not recognised, please type it in carefully')
                movie_ = input()
                row = movies_df[movies_df['title'] == movie_]
                
            movie_id = list(row['movie_id'])[0]

            imdb_link = movie_url[movie_url['movie_id'] == int(movie_id)]['link'].values[0]
            
            try:
                print('The IMDB link is: ' + imdb_link + ' where additional information such as movie description can be found.')
            except:
                print('Sorry the IMDB link cannot be found for this movie')
            
            flag = False
        
    
    
    

elif x == 'U':
    print('Please be aware that any data you add or edit will be stored anonymously')
    print('Please select the appropriate option')
    print('R - Edit User Reviews')
    print('X - Exit')
    
    x = input()
    accepted = check_if_valid_input(x, ['R', 'C', 'X'])
    while accepted == False:
        print('That is not a valid option. Please choose from the existing options')
        x = input()
        accepted = check_if_valid_input(x, ['R', 'C', 'X'])
    x = remove_whitespace_and_capitalise(x)
    
    if x == 'X':
        quit() 
    
    elif x == 'R':
        print('Please select the appropriate option')
        print('A - Add a rating')
        print('E - Edit a rating')
        print('R - Remove a rating')
        
        x = input()
        accepted = check_if_valid_input(x, ['A', 'E', 'R'])
        while accepted == False:
            print('That is not a valid option. Please choose from the existing options')
            x = input()
            accepted = check_if_valid_input(x, ['A', 'E', 'R'])
        x = remove_whitespace_and_capitalise(x)
        model = keras.models.load_model('model_RNN')
        
        if x == 'A':
            new_training_df = pd.DataFrame(columns = ['user_id', 'movie_id', 'rating', 'timestamp'])
            while True:
                print('Please enter Movie ID from 1 to ' + str(num_movies) + ':')
                movie_id = input()
                accepted = check_if_valid_movie_id(movie_id , num_movies)
                while accepted == False:
                    print('Please enter a valid movie ID')
                    movie_id  = input()
                    accepted = check_if_valid_movie_id(movie_id , num_movies)
                
                print('Please enter rating between 1 and 5')
                rating = input()
                accepted = check_if_valid_rating(rating)
                while accepted == False:
                    print('Please enter a valid rating')
                    rating  = input()
                    accepted = check_if_valid_rating(rating)
                    
                if int(user_id) > 6040:
                    user_id = 6040
                new_training_df = new_training_df.append({'user_id': user_id, 'movie_id':movie_id, 'rating': rating, 'timestamp': timestamp +1}, ignore_index=True)
                
                print('Would you like to add another rating')
                print('Y - yes')
                print('N - no')
                
                x = input()
                accepted = check_if_valid_input(x, ['Y', 'N'])
                while accepted == False:
                    print('That is not a valid option. Please choose from the existing options')
                    x = input()
                    accepted = check_if_valid_input(x, ['Y', 'N'])
                x = remove_whitespace_and_capitalise(x)
                if x == 'N':
                    break 
            

            
        elif x == 'R': 
            while True: 
                print('Please enter Movie ID from 1 to ' + str(num_movies) + ':')
                movie_id = input()
                accepted = check_if_valid_movie_id(movie_id , num_movies)
                while accepted == False:
                    print('Please enter a valid movie ID')
                    movie_id  = input()
                    accepted = check_if_valid_movie_id(movie_id , num_movies)
                    
                row = ratings_df[(ratings_df['movie_id'] == int(movie_id)) & (ratings_df['user_id'] == int(user_id))]
                
                if len(row) == 0:
                    print('Sorry user ' + user_id +' has not rated that movie')
                
                else:
                    index = row.index
                    ratings_df = ratings_df.drop(index = index)
                    print('Successfully removed')
                    
                print('Would you like to remove another rating')
                print('Y - yes')
                print('N - no')
                
                x = input()
                accepted = check_if_valid_input(x, ['Y', 'N'])
                while accepted == False:
                    print('That is not a valid option. Please choose from the existing options')
                    x = input()
                    accepted = check_if_valid_input(x, ['Y', 'N'])
                x = remove_whitespace_and_capitalise(x)
                if x == 'N':
                    break 
                
        elif x == 'E':
            while True: 
                print('Please enter Movie ID from 1 to ' + str(num_movies) + ':')
                movie_id = input()
                accepted = check_if_valid_movie_id(movie_id , num_movies)
                while accepted == False:
                    print('Please enter a valid movie ID')
                    movie_id  = input()
                    accepted = check_if_valid_movie_id(movie_id , num_movies)
                
                row_len = len(ratings_df[(ratings_df['movie_id'] == int(movie_id)) & (ratings_df['user_id'] == int(user_id))])
                
                if row_len == 0:
                    print('Sorry user ' + user_id +' has not rated that movie')
                    
                else:
                    print('Please enter rating (between 1 and 5)')
                    rating = input()
                    accepted = check_if_valid_rating(rating)
                    while accepted == False:
                        print('Please enter a valid user ID')
                        rating  = input()
                        accepted = check_if_valid_rating(rating)
                    
                    
                    if int(user_id) > 6040: 
                        user_id = 6040
                    new_training_df = new_training_df.append({'user_id': user_id, 'movie_id':movie_id, 'rating': rating, 'timestamp': timestamp +1}, ignore_index=True)
                    
                print('Would you like to edit another rating')
                print('Y - yes')
                print('N - no')
                
                x = input()
                accepted = check_if_valid_input(x, ['Y', 'N'])
                while accepted == False:
                    print('That is not a valid option. Please choose from the existing options')
                    x = input()
                    accepted = check_if_valid_input(x, ['Y', 'N'])
                x = remove_whitespace_and_capitalise(x)
                if x == 'N':
                    break 
        
        #transfer learning
        print('Updating User Profile... ')
        if len(new_training_df) != 0:
            

            x = list(new_training_df['user_id'])
            y = list(new_training_df['movie_id'])
            rating = list(new_training_df['rating'])
           
            
            x = list(map(int, x))
            y = list(map(int, y))
            rating = list(map(int, rating))

        
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', run_eagerly=True )
            model.fit([np.array(x),np.array(y)], np.array(rating), epochs=1, batch_size=32, verbose=0)
            
            model.save('model_')
            
            ratings_df = ratings_df.append(new_training_df)
            ratings_df.to_csv('ratings_updated.csv')
            
        print('Updated User Profile')
    
