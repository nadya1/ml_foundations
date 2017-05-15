__author__ = 'nadyaK'
__date__ = '04/14/17'

import ml_graphlab_utils as gp
import traceback

def select_most_used_word(products, selected_words, words_count):
	most_used = {}
	for selected_wd in selected_words:
		products[selected_wd] = words_count.apply(lambda x:selected_wd in x)
		most_used[selected_wd] = products[selected_wd].sum()

	return products, most_used

def main():
	try:
		#Load Data
		products = gp.load_data('../../data_sets/amazon_baby.gl/')

		#Create word-count column
		products['word_count'] = gp.get_text_analytics_count(products['review'])

		#Select a group of words
		selected_words = ['awesome','great','fantastic','amazing','love','horrible',
						  'bad','terrible','awful','wow','hate']

		# Q1: Out of the 11 words in selected_words, which one
		# is most used in the reviews in the dataset?
		products,most_used = select_most_used_word(products, selected_words, products['word_count'])

		key_most_used = gp.find_key_max(most_used)
		least_most_used = gp.find_key_min(most_used)
		print "\nQ1: Most used:%s = %s" % (key_most_used, most_used[key_most_used])
		print "\nQ2: Least used:%s = %s" % (least_most_used, most_used[least_most_used])

		#ignore all 3* reviews (to remove unknown rating)
		products = products[products['rating'] != 3]

		#positive sentiment = 4* or 5* reviews
		products['sentiment'] = products['rating'] >= 4

		#Split Data
		train_data,test_data = products.random_split(.8,seed=0)

		#***************
		# Create Model *
		#***************
		#Create Logistic Model (with selected-words as features)
		selected_model = gp.create_logistic_classifier_model(train_data,target='sentiment',
													   features=selected_words,
													   validation_set=test_data)
		#Create Logistic Model (with word_count as features)
		sentiment_model = gp.create_logistic_classifier_model(train_data,target='sentiment',
														features=['word_count'],
														validation_set=test_data)

		# Get weights of Coefficients
		coefficients = selected_model['coefficients'].sort('value',ascending=False)
		# print coefficients.print_rows(12)

		#Out of the 11 words in selected_words, which one got
		# the most positive/negative weight in the selected_words_model
		print gp.find_key_max({coefficients['name']:coefficients['value']})
		print "\nQ3: Most Positive (w): love"
		print "\nQ4: Most Negative (w): terrible"

		#*****************
		# Evaluate Model *
		#*****************
		#Which of the following ranges contains the accuracy
		# of the selected_words_model on the test_data
		results_selected = selected_model.evaluate(test_data)
		results_sentiment = sentiment_model.evaluate(test_data)
		print "\nQ5: Accuracy selected-model: %s" %results_selected['accuracy'] #0.843111938506
		print "\nQ6: Accuracy sentiment-model: %s" %results_sentiment['accuracy'] #0.916256305549

		#****************
		# Predict Model *
		#****************
		#Which of the following ranges contains the predicted_sentiment for the most positive review
		# for Baby Trend Diaper Champ, according to the sentiment_model ?
		diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']
		diaper_champ_reviews['predicted_selected'] = selected_model.predict(diaper_champ_reviews,
																			 output_type='probability')
		diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews,
																			 output_type='probability')

		most_positive_review = diaper_champ_reviews.sort('predicted_selected',ascending=False)
		most_positive_sentiment = diaper_champ_reviews.sort('predicted_sentiment',ascending=False)

		print "\nQ9: predicted_selected most positive review: %s"%max(most_positive_review['predicted_selected'])
		print "\nQ10: predicted_sentiment most positive review: %s"%max(most_positive_sentiment['predicted_sentiment'])

	except Exception as details:
			print "Error >> %s" % details
			traceback.print_exc()

if __name__ == "__main__":
	main()