__author__ = 'nadya1'
__date__ = "04/01/2017"

import ml_graphlab_utils as gp

class ML_Foundations(object):
	def __init__(self):
		self.course_name ='ML Foundations'
		self.gp = gp # Graphlab utils

	def create_linear_regression(self, dataset, target, features, validation_set=None, verbose=False):
		return gp.create_linear_regression(dataset, target, features, validation_set, verbose)

	def create_logistic_classifier_model(self, dataset, target, features, validation_set=None, verbose=False):
		return gp.create_logistic_classifier_model(dataset,target,features,validation_set,verbose)

	def create_nearest_neighbors_model(self, dataset, features, label, distance=None, verbose=False):
		return gp.create_nearest_neighbors_model(dataset,features=features,label=label,
												  distance=distance,verbose=verbose)

	def create_popularity_recommender_model(self, dataset, user_id, item_id, verbose=False):
		return gp.create_popularity_recommender_model(dataset, user_id, item_id, verbose)

	def create_similarity_recommender_model(self, dataset, user_id, item_id, verbose=False):
		return gp.create_similarity_recommender_model(dataset, user_id, item_id, verbose)

	def make_model_predictions(self, model, SFrame):
		"""Make predictions dataset : SFrame | pandas.Dataframe"""
		return model.predict(SFrame)

	def evaluate_model(self, model, dataset):
		"""Evaluate results"""
		return model.evaluate(dataset)

	def get_coefficients(self, model):
		return model.get('coefficients')

def main():

	# Week_1
	# Create/modify new columns in  SFrame
	sframe = gp.load_data('../data_sets/people-example.csv')
	# print sframe.show()
	print sframe.tail()
	# sframe['Country'] = sframe['Country'].apply(lambda x: 'United States' if x == 'USA' else x)
	sframe = gp.transform_column_entry(sframe,'Country','USA','United States')
	print 'New SFrame:\n',sframe

if __name__ == "__main__":
	main()
