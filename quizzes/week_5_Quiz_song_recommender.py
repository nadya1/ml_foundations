__author__ = 'nadyaK'
__date__ = '04/20/17'

import ml_graphlab_utils as gp
import traceback

def week5_summary(song_data):
	#Count number of unique users in the dataset
	users = song_data['user_id'].unique()

	#Split data
	train_data,test_data = song_data.random_split(.8,seed=0)

	#Simple popularity-based recommender
	#A popularity model makes the same prediction for all users, so provides no personalization.
	popularity_model = gp.create_popularity_recommender_model(train_data,user_id='user_id',item_id='song')
	print popularity_model

	#Build a song recommender with personalization
	#personalized recommendations to each user.
	personalized_model = gp.create_similarity_recommender_model(train_data,user_id='user_id',item_id='song')
	print personalized_model

def counting_unique_users(song_data, list_users, filter_col='artist', unique_by='user_id'):
	# find out the number of unique users
	unique_users = {}
	for user in list_users:
		filter_user= song_data[song_data[filter_col]==user]
		unique_users[user] = len(filter_user[unique_by].unique())
	return unique_users

def main():
	try:
		song_data = gp.load_data('../../data_sets/song_data.gl/')

		artist_list = ['Kanye West', 'Foo Fighters', 'Taylor Swift', 'Lady GaGa']
		count_uniques = counting_unique_users(song_data, artist_list)
		# print count_uniques

		#Which of the artists below have had the most unique users listening to their songs
		print "\nQ1: Most unique users: %s" % (gp.find_key_max(count_uniques))

		#Which of the artists below is the most popular artist,
		# the one with highest total listen_count, in the data set
		listen_count = song_data.groupby(key_columns='artist',
										operations={'total_count': gp.graphlab.aggregate.SUM('listen_count')})

		# print listen_count.sort('total_count',ascending=False) #most listend / ascending=True) #least listend
		most_listen_count = gp.convert_sframe_to_simple_dict(listen_count, 'artist', 'total_count')
		print "\nQ2: Highest total listen: %s" % (gp.find_key_max(most_listen_count))
		print "\nQ3: Smallest total listen: %s" % (gp.find_key_min(most_listen_count))

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()
