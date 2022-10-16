# Producing Diverse Movie Recommendations using Deep Model-Based Recommender Systems 
Submitted as part of the degree of Bsc Natural Sciences to the Board of Examiners in the Department of Computer Sciences, Durham University. 
This summative assignment was assessed and marked by the professor of the module in question:
## Grade: 1st - 89/100, 3rd in year (of 116 students).
## Paper Introduction:
We implement two state-of-the-art deep model-based recommender systems (RS), a novel content-collaborative hybrid RS and a solely collaborative RS. These RSs foster user satisfaction in different ways; the first by capturing the complex user-item interaction structures and the second that exploits temporal data. To compliment these RSs we create a fully-fledged UI where user can log in, register, submit new reviews and retrieve personalized movie recommendations using our two recommender systems. Due to being highly optimized, the recommender can retrain itself almost instantly when a user submits a new review or on alteration to a prediction which allows the user to interact with the recommender and see the recommendations change in real-time. 

We then contrast the two different recommender methodologies and evaluate which is most effective with respect to customer satisfaction and engagement - metrics conducive to profit. We also comment on which feedback form (explicit or implicit) is more preferable to exploit from the perspective of our results.

## Contents of repo:
* readme.txt - Text file outlining how to run the recommender systems and command-line interface
* pretrained_models - Pretrained models trained on a NVDIA GPU
* paper.pdf - Report outling methodology and results 
* recommender_system.gif - A demo video of the command-line interface allowing the user to interact with the Recommender Systems
* metrics_novelty_NDCG.py - python file to calculate the novelty and diversity of the recommendations
* preprocessing_to_create_user_and_movie_vectors.py - python file that gathers additional information on movies from IMBD and processes this into vector form (by passing the poster through ResNet50 and the description through BERT and taking the tensor product to combine these two distinct vector spaces, then various feature selection methods were performed to decrease the dimensionality) 
* RNN.py - my implementation of [Recurrent Recommender Networks](https://alexbeutel.com/papers/rrn_wsdm2017.pdf)
* NeuMF_hybrid.py - my implementation of [NeuMF](https://arxiv.org/pdf/1708.05031.pdf)(includes the novel contribution of content-based item and user vectors)
* main.py - python file to run the command-line interface and recommender systems

## Demo video 
  > ![Gifdemo](https://github.com/Lauren-Stumpf/Recommender_Systems_Coursework/blob/main/recommender_system.gif)
  > 

