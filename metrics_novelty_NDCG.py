

import pandas as pd 
import math 
from sklearn.metrics import ndcg_score
import numpy as np
import random 


#Read in datasets
ratings_df = pd.read_csv('ratings.csv')
user_item_matrix_predicted_ratings_RNN = pd.read_csv('user_item_RNN.csv')
user_item_matrix_predicted_ratings_NeuMF = pd.read_csv('user_item_NeuMF.csv')


#Define functions

#Functions for NDCG
def return_true_relevence(user_id):
    
    user_df = ratings_df[ratings_df['user_id'] == user_id]
    user_df = user_df.sort_values(by=['rating'], ascending = False)

    return list(user_df['rating'])


def return_relevance_score(user_id, user_item_matrix):

    user_df = ratings_df[ratings_df['user_id'] == user_id]
    list_that_user_has_rated = list(user_df['movie_id'])
    indices = list(user_item_matrix.index)

    list_that_user_has_rated = [i for i in list_that_user_has_rated if i in indices]

    temp_df = user_item_matrix[user_id].sort_values(ascending = False)
    temp_df = temp_df.loc[list_that_user_has_rated]
    movies_in_order = temp_df.index
    
    ordered = []
    for movie in movies_in_order:
        rating = user_df[user_df['movie_id'] == movie]['rating'].values[0]
        ordered.append(rating)

    return ordered

#Function for novelty 
def return_proportion(item, number_of_recommendations):
    
    subset_ratings = len(ratings_df[ratings_df['movie_id'] == item])
    proportion = subset_ratings/number_of_recommendations
    
    return proportion




recommender_systems = [ user_item_matrix_predicted_ratings_NeuMF, user_item_matrix_predicted_ratings_RNN]

count = 1
for user_item_matrix_predicted_ratings in recommender_systems:
    print('Recommender System: ' + str(count))
    count += 1
    user_item_matrix_predicted_ratings = user_item_matrix_predicted_ratings.drop(columns = ['Unnamed: 0'])
    user_item_matrix_predicted_ratings = user_item_matrix_predicted_ratings.T
    user_item_matrix_predicted_ratings.columns = range(1,6041)
    user_item_matrix_predicted_ratings.index = range(1, 3884)
    
    
    #Calculate novelty
    
    novelty = []
    user_ids = [random.randint(1, 6040) for i in range(50)]
    for user_id in user_ids: 
        R = 25
        predictions = user_item_matrix_predicted_ratings.sort_values(by=[user_id], ascending = False)[user_id].index
        list_of_predictions = list(predictions)[:R]
        
        sum_ = 0
        for item in list_of_predictions: 
            p = return_proportion(item, R)
            
            #Avoid domain error 
            if p != 0:
                sum_ += -math.log(p,2)
            else:
                R = R -1
    
        sum_ /= R 
        novelty.append(sum_)
    
    print('Novelty')
    print(sum(novelty) / len(novelty))
    
    #NDCG
    
    ndcg_list = []
    for user_id in user_ids: 
    
        num_items = 10
    
        # Relevance scores in output order
        relevance_score = np.asarray([return_relevance_score(user_id, user_item_matrix_predicted_ratings)])
        
        # Relevance scores in Ideal order
        true_relevance = np.asarray([return_true_relevence(user_id)])
        
        try: 
            ndcg = ndcg_score(true_relevance, relevance_score, k =10)
            ndcg_list.append(ndcg)
        except: #Wrong shape, so reshape and try again  
            pass
        
    print('NDCG')
    print(sum(ndcg_list) / len(ndcg_list))





"""
These experiments were repeated over three random seeds, with three different models for each recommender system, each being trained on a different train-test split. 
There was not space to include all three separately trained models however the results can be found in my report. This is the result for the first random seed (1). 
"""