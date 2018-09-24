from kumparanian import ds

class Model(object):
	def __init__(self):
		import pandas as pd

		#read dataset
		data_frame = pd.read_csv('data.csv')
		#clean null values if any
		data_frame = data_frame[pd.notnull(data_frame['article_content'])]
		#only use article_topic and article_content columns
		cols = ['article_topic', 'article_content']
		data_frame = data_frame[cols]
		data_frame.columns = ['article_topic', 'article_content']

		#make categorization upon article topic, ex: international = 1, sepakbola = 2, ...
		data_frame['article_category'] = data_frame['article_topic'].factorize()[0]

		#create category2index
		from io import StringIO
		#createa category_id data frame by removing all duplicates from article_topic, article_category
		category_id_data_frame = data_frame[['article_topic', 'article_category']].drop_duplicates().sort_values('article_category')
		category_to_id = dict(category_id_data_frame.values)
		id_to_category = dict(category_id_data_frame[['article_category', 'article_topic']].values)

		self.id_to_category = id_to_category
		'''
		if you want to train my model using whole dataset
		you need to change code below:
		self.data_frame = data_frame
		'''
		self.data_frame = data_frame[0 : 6000]

		'''
		feature extraction from text
		'''
		from sklearn.feature_extraction.text import TfidfVectorizer

		self.tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', 
								ngram_range = (1, 2), stop_words = 'english')

		'''
		our transformed features and labels
		'''
		self.features = self.tfidf.fit_transform(self.data_frame.article_content).toarray()
		self.labels = self.data_frame.article_category

	def train(self):
		#load our classifier, Linear Support Vector Classifier
		from sklearn.svm import LinearSVC

		#initialize our model
		self.model = LinearSVC()

		#train our model
		self.model.fit(self.features, self.labels)

		#save trained model
		self.save()

	def predict(self, input_text):
		
		input_text = [input_text] #only bracket input accepted
		#transform input text
		text_features = self.tfidf.transform(input_text)
		#predict
		article_topic_predicted = self.model.predict(text_features)

		#return the article_topic
		return self.id_to_category[article_topic_predicted[0]]

	def save(self):
		import pickle
		
		my_model = self.model
		my_pickle_file = 'model.pickle'
		outfile = open(my_pickle_file, 'wb')
	
		#dump
		pickle.dump(my_pickle_file, outfile)
		outfile.close()
		

		ds.model.save(self, "model.pickle")


if __name__ == '__main__':
    # NOTE: Edit this if you add more initialization parameter
    model = Model()

    # Train your model
    model.train()

    # Save your trained model to model.pickle
    model.save()