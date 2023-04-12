# Data-Science-Capstone-Project

Capstone project for the Concordia Data Science bootcamp

## Topic Modeling of Headlines From The New York Times 2013-2023

### With an emphasis on exploring climate change coverage

A subset of article headlines and abstracts gathered from the New York Times api 
were used to create a topic model seeded with climate change related terms. This
model was used to explore how topics related to climate change have been covered
by the New York Times over time. The initial plan was to pull data from multiple
newspapers but there was a challenge with being able to run the topic modelling 
within memory constraints. 

The model was created using a pretrained sentence transformer (all-mpnet-base-v2)
on 60% of the total data that was obtained for the period of January 1st 2013 to 
December 31st 2022. The colab notebook and information on the pretrained models 
available are below. The New York Times data as well as the model and embeddings
are available on Google Drive for download [here](https://drive.google.com/drive/folders/1OjSaMasVDKDQd1QaceANRktCisLXCMEH?usp=share_link).



[Colab Notebook](https://colab.research.google.com/drive/1oj5u-RSKqfMgWVzWYfKZ_ZhV59LolF-d?usp=sharing)

[List of pretrained models](https://www.sbert.net/docs/pretrained_models.html)