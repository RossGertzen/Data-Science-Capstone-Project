import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from bertopic import BERTopic
from nrclex import NRCLex

### PAGE SETUP ###
st.set_page_config(layout='wide', page_icon=':bar_chart:')
# st.cache_data.clear()
	
### ANALYSIS INFO ###
buff, col, buff = st.columns([1,4,1])

with col:
	st.title('Topic Modeling of Headlines From The New York Times 2013-2023')

	st.header('With an emphasis on exploring climate change coverage')

	st.write("""A subset of article headlines and abstracts gathered from the New York Times api 
				were used to create a topic model seeded with climate change related terms. This
				model was used to explore how topics related to climate change have been covered
				by the New York Times over time. The initial plan was to pull data from multiple
				newspapers but there was a challenge with being able to run the topic modelling 
				within memory constraints. 
				""")
	st.write("""The model was created using a pretrained sentence transformer (all-mpnet-base-v2)
				on 80% of the total data that was obtained for the period of January 1st 2013 to 
				December 31st 2022. The colab notebook and information on the pretrained models 
				are available below. The New York Times data as well as the model and embeddings
				are available on Google Drive for download [here](https://drive.google.com/drive/folders/1OjSaMasVDKDQd1QaceANRktCisLXCMEH?usp=share_link).
				""")
	st.write("""Unfortunately the model that is used here is really bad. There were endless issues
				with environment compatability between google colab and my local environment. There
				are far too many outliers and small topics here. I was able to get better models but
				they could not be loaded into the local environment. In the future, I would avoid 
				using google colab and find another way to access GPU runtime. Zero stars.
				""")
	st.write('[Colab Notebook](https://colab.research.google.com/drive/1oj5u-RSKqfMgWVzWYfKZ_ZhV59LolF-d?usp=sharing)')
	st.write('[List of pretrained models](https://www.sbert.net/docs/pretrained_models.html)')
	
st.write("---")
st.write("##")


### FUNCTIONS ###

# cache the model that gets loaded because it can take awhile
@st.cache_resource(show_spinner='Loading Model')
def get_topic_model(model_file, embedding_model='all-mpnet-base-v2'):
	topic_model = BERTopic.load(model_file, embedding_model=embedding_model)

	return topic_model

# load embeddings to use with the topic model
@st.cache_resource(show_spinner='Loading Model')
def get_embeddings(embedding_file):
	embeddings = np.load(embedding_file)
	
	return embeddings
	
# load in nyt data
@st.cache_data(persist=True, show_spinner='Loading Data')
def get_nyt():
	nyt = pd.read_csv('./nyt/nyt_data.csv')
	nyt['pub_date'] = pd.to_datetime(nyt['pub_date'])

	return nyt

# create topic labels with all 10 words per topic
@st.cache_data(persist=True)
def make_top_word_labels(_topic_model):
	top_word_labels = dict(zip(list(range(-1,len(topic_model.custom_labels_))), topic_model.custom_labels_))
	
	return top_word_labels

# create table based on search input
def topic_search_df(find_topic):
	df = pd.DataFrame(topic_model.find_topics(find_topic), index=['Topic', 'Prob.']).T
	df.Topic = df.Topic.astype(int)
	df['Top 10 words in topic'] = df['Topic'].map(top_word_labels)
	
	return df

def plot_topic_frequencies(df, topic_model, topics_of_interest, period='Q', forecast=True):
	"""
	Create a series of plots showing the frequency of each
	topic given.
	
	Parameters
	----------
	df: dataframe with 'text' and 'pub_date' columns
	topic_model: BERTopic model with embeddings tied
					to dataframe
	topics_of_interest: list of topic numbers to plot
	period: the desired period frequency to plot
	forecast: bool whether to include the forecast
	"""
	# manually calculate the frequency of each topic among all others within that period
	df_ = pd.DataFrame({'docs': df.text, 'topic': topic_model.topics_, 'period': df.pub_date.dt.to_period(period)})
	df_['topic_labels'] = df_['topic'].map(topic_model.topic_labels_)
	df_['freq'] = (df_.groupby(['period', 'topic'])['topic'].transform('count')
				   /df_.groupby('period')['topic'].transform('count'))
	
	# clean up calculated and transformed table so that only one value for each period for topics of interest
	df_freqs = df_[['period', 'topic', 'topic_labels', 'freq']][df_.topic.isin(topics_of_interest)]
	df_freqs.drop_duplicates(inplace=True)

	# build a period index for timeseries
	df_freqs = df_freqs.set_index('period').sort_index()
	df_freqs = df_freqs[['topic_labels', 'freq']].pivot(columns=['topic_labels'])
	
	# make sure every period is accounted for even if there is no data for that topic
	idx = pd.date_range('2013Q1', '2023Q1', freq=period).to_period()
	df_freqs = df_freqs.reindex(idx, fill_value=0)
	df_freqs.fillna(0, inplace=True)
	
	# perform a quick and dirty ARIMA model if requested
	if forecast:
		preds = pd.DataFrame()
		for col in df_freqs:
			arima = auto_arima(df_freqs[col], max_d=3)
			order = arima.get_params()['order']
			model = ARIMA(df_freqs[col], order=order).fit(method_kwargs={'maxiter':1000})
			pred = model.predict(start=len(df_freqs[col]), end=len(df_freqs[col])+2, typ='levels').rename(df_freqs[col].name)
			preds = pd.concat([preds, pred], axis=1, copy=False)
		# add the predictions to the created table
		df_freqs = pd.concat([df_freqs, preds], axis=0, copy=False)
	
	# build the plot
	axes = df_freqs.plot.line(y='freq', grid=True, alpha=0.9, subplots=True, sharey=True, ylabel='Frequency',
							  sharex=True, title='Topic Frequency per Period', figsize=(10,10))
	for ax in axes:
		ax.axvline(pd.to_datetime('2023Q1'), color='r', linestyle='--')
	
	plt.title('Topic Frequency per Period')
	return plt.gcf()
	
def plot_topic_emotion_profile(df, topic_model, topic, counts=False, over_time=False, sentiment_only=False):
	"""
	Generate a bar chart to view sentiment/emotion
	profile of the given topic.
	
	Parameters
	----------
	df: dataframe with 'text' and 'pub_date' columns
	topic_model: BERTopic model with embeddings tied
					to dataframe
	counts: bool default False
					False for frequencies
					True for raw counts
	over_time: bool default False
					False for all years combined
					True for year by year
	sentiment_only: bool default False
					False for all emotions
					True for pos and neg only
	"""
	# create dataframe with relevant information
	df_ = pd.DataFrame({'docs': df.text, 'topic': topic_model.topics_, 'timestamp': df.pub_date}) 
	# instatiate the dataframe to plot from
	profile = pd.DataFrame()
	
	# words for titles or columns
	method = 'Counts' if counts else 'Frequencies'
	timeframe = 'by Year' if over_time else '2013-2023'
	
	# split emotion profile by year and create dataframe for plotting
	if over_time:   
		for year in range(2013,2023):
			text = ' '.join(df_[(df_.topic == topic) & (df_.timestamp.dt.year == year)]['docs'])
			text_object = NRCLex(text)
			if counts:
				data = text_object.raw_emotion_scores
			else:
				data = text_object.affect_frequencies
			emote_df = pd.json_normalize(data).T
			emote_df.columns = [year]
			emote_df = emote_df.sort_values(year, ascending=False)
			profile = pd.concat([profile, emote_df], axis=1, copy=False)
			
	# create emotion dataframe without splitting by year
	else:
		text = ' '.join(df_[df_.topic == topic]['docs'])
		text_object = NRCLex(text)
		if counts:
			data = text_object.raw_emotion_scores
		else:
			data = text_object.affect_frequencies
		emote_df = pd.json_normalize(data).T
		emote_df.columns = [method]
		profile = emote_df.sort_values(method, ascending=False)
		
	# clean up plotting dataframe
	profile.fillna(0, inplace=True)
	if counts:
		profile = profile.astype(int, copy=False)
	if sentiment_only:
		profile = profile.loc[['positive', 'negative']]

	# plot profile dataframe as bar chart
	fig, ax = plt.subplots(figsize=(10,5))
	profile.plot.bar(title=f'Emotion Profile {method} for Topic {topic_model.get_topic_info(topic)["Name"][0]} {timeframe}',
					width=0.8, ax=ax)
	return fig


### DATA PREP ###

df = get_nyt()

# must use same state as was used to create model
df_reduced = df.sample(frac=0.8, random_state=10)

#texts_reduced = list(df_reduced['text'])

# fetch the model and embeddings
topic_model= get_topic_model('./models/all_mpnet_model_80_1')
embeddings = get_embeddings('./embedding_models/all_mpnet_embeddings_80_1.npy')

# topics as a list for all documents
topics = topic_model.topics_

# create a custom set of topic labels that include top 10 words
topic_labels = topic_model.generate_topic_labels(nr_words=10, topic_prefix=True, word_length=15, separator=" ")
topic_model.set_topic_labels(topic_labels)

top_word_labels = make_top_word_labels(topic_model)


### INTERACTIVE ELEMENTS ###

st.sidebar.write('All topics:')

topic_names = pd.DataFrame(topic_model.topic_labels_.values())
topic_names = pd.concat([topic_names, topic_model.get_topic_freq().sort_values('Topic')], axis=1)
topic_names.columns = ['Topic Names','Topic','Count']
topic_names.drop('Topic', axis=1, inplace=True)

st.sidebar.dataframe(topic_names, height=700)

st.title('Explore Topics')
st.write('Investigate the topics generated with BERTopic.')
st.write('[BERTopic Repo](https://github.com/MaartenGr/BERTopic)')
	
with st.container():
	buff, col1, buff, col2, buff = st.columns([0.25,3,0.25,4,1])

	if 'search_df' not in st.session_state:
		st.session_state.search_df = topic_search_df('')

	if 'interest_topics' not in st.session_state:
		st.session_state.interest_topics = ''

	def topic_search():
		st.session_state.search_df = topic_search_df(search_topic)

	def topic_viz():
		st.session_state.interest_topics = [int(i) for i in topics_of_interest.split(',')]

	with col1:
		st.header('Start Here')
		st.write("""Find topic numbers with terms or phrases.
		Topics will be found based on cosine similarity using the
		BERTopic library. Try: 'climate change', 'pollution'.""")
		
		search_topic = st.text_input('Search for a Topic:')

		if search_topic:
			topic_search()
			st.dataframe(st.session_state.search_df)


	with col2:
		st.header('Cluster Visualization')
		st.write('Cannot finish with large datasets')
	
		topics_of_interest = st.text_input('Enter topic numbers separated by commas to visualize 2D representation of clusters')

		if topics_of_interest:
			topic_viz()
			st.plotly_chart(topic_model.visualize_documents(list(df_reduced['text']), 
															embeddings=embeddings, 
															sample=0.0001, 
															hide_annotations=True, 
															hide_document_hover=True, 
															topics=st.session_state.interest_topics, 
															custom_labels=True ))
			
st.write("---")
st.write("##")

# Emotion/sentiment analysis section
with st.container():
	
	st.title('Emotion/Sentiment Analysis Using NRCLex library')
	st.write('Inspect the emotion profile of a topic based on the Emotion Lexicon from National Research Council Canada.')
	st.write('[See NRCLex Repo](https://github.com/metalcorebear/NRCLex)')
	
	# create columns for form and chart
	col3, col4 = st.columns([1,4])

	# lookups for display conversion
	counts_values = {
		False: 'Frequencies',
		True: 'Counts'}

	over_time_values = {
		False: 'Combined',
		True: 'By Year'}

	sentiment_only_values = {
		False: 'All Emotions',
		True: 'Sentiment Only'}
	
	# methods for display conversion
	def convert_counts(option):
		return counts_values[option]

	def convert_over_time(option):
		return over_time_values[option]

	def convert_sentiment_only(option):
		return sentiment_only_values[option]
	
	# make the form to get parameters for bar chart graphing function
	with col3:
		with st.form('emotion_profile'):
			topic_num = st.number_input('Enter Topic Number', value=0, format='%d')
			counts = st.selectbox('Metric', [True, False], format_func=convert_counts)
			over_time = st.selectbox('Timeframe', [True, False], format_func=convert_over_time)
			sentiment_only = st.selectbox('Emotion Detail', [True, False], format_func=convert_sentiment_only)

			profile = st.form_submit_button('Create Bar Chart')

	# plot the chart after the button has been pressed
	with col4:
		if profile:
			bar_plot = plot_topic_emotion_profile(df_reduced, topic_model, topic_num, counts, over_time, sentiment_only)
			st.pyplot(bar_plot)
			
st.write("---")
st.write("##")

# Topic frequency over time section
with st.container():
	
	st.title('Graph Topic Frequency Over Time')
	st.write('Frequencies calculated based on period selected.')
	st.write('A Statsmodels ARIMA model is used to estimate the forecast.')
	
	col5, col6 = st.columns([1,4])

	# conversion lookup
	period_values = {
		'Q': 'Quarter',
		'Y': 'Year',
		'M': 'Month'}
	
	# conversion method
	def convert_period(option):
		return period_values[option]

	# get parameters for frequency over time plot
	with col5:
		with st.form('topics_over_time'):
			topics_for_plot = st.text_input('Topic Numbers to Plot\n(More than 6 not recommended)', )
			period = st.selectbox('Select Period Freqency', ['Q', 'Y', 'M'], format_func=convert_period)
			forecast = st.checkbox('Include Forecast')

			profile = st.form_submit_button('Create Graphs')

	# plot if button pressed
	with col6:
		if profile:
			topics_for_plot = [int(i) for i in topics_for_plot.split(',')]
			line_plot = plot_topic_frequencies(df_reduced, topic_model, topics_for_plot, period, forecast)
			st.pyplot(line_plot)
			
st.write('##')
st.write('##')		
	
buff, foot, buff = st.columns([1,2,1])			

with foot:
	st.write('Created for the Concordia Data Science Capstone Project April 2023 :copyright: Ross Gertzen :mending_heart::earth_africa:')
	st.write('Huge thanks to the instructors and cohort who helped make this possible!')