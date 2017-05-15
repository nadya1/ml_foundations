__author__ = 'nadyaK'
__date__ = '04/18/17'

import ml_graphlab_utils as gp
import traceback

def stack_columns_to_table(SFrame, col_name, new_cols, sort_by='count', ascending=False):
	word_count_table = SFrame[[col_name]].stack(col_name,new_column_name=new_cols).sort(sort_by,ascending=ascending)
	return word_count_table

def calculate_cos_distance(src_name, names_to_compare, SFrames, compare_by='tfidf'):
	cos_distance = {}
	for name in names_to_compare:
		SFrame_src = SFrames[src_name][compare_by][0]
		SFrame_compare = SFrames[name][compare_by][0]
		cos_distance['%s_vs_%s'%(src_name,name)] = gp.get_cosine_distance(SFrame_src,SFrame_compare)
	return cos_distance

def query_min_knn_distance(name, people_dist, knn_model, name_model, count_qs):
	raw_model = knn_model.query(people_dist[name],verbose=False)
	# print raw_words
	raw_dict = gp.convert_sframe_to_simple_dict(raw_model,'reference_label','distance')
	raw_dict.pop(name)
	print "\nQ%s: %s: (%s): %s" % (count_qs,name,name_model,gp.find_key_min(raw_dict))

def main():
	try:
		people = gp.load_data('../../data_sets/people_wiki.gl/')

		#Create Word Count & TF_IDF analytics count
		people['word_count'] = gp.get_text_analytics_count(people['text'])
		people['tfidf'] = gp.get_text_analytics_tf_idf(people['word_count'])

		famous_people = ['Elton John','Victoria Beckham','Paul McCartney'] #Quiz
		# famous_people = ['Barack Obama', 'Bill Clinton', 'David Beckham', 'Taylor Swift', 'George Clooney']
		people_info = {}

		for person in famous_people:
			people_info[person] = people[people['name'] == person]
			people_info['%s table'%person] = stack_columns_to_table(people_info[person], 'word_count', ['word','count'])
			people_info['%s tfidf'%person] = stack_columns_to_table(people_info[person], 'tfidf', ['word','tfidf'], sort_by='tfidf')


		# 1)Person:'Elton John' What are the 3 words in his articles
		# with highest word counts? and  with highest TF-IDF?
		name = 'Elton John'
		print "Person: %s"%name
		print "\nQ1: Highest word counts = %s" % (people_info['%s table'%name])
		print "\nQ2: Top TF-IDF= %s" % (people_info['%s table'%name])

		# 2)Whats the cosine distance between the articles on
		dist1 ='Elton John_vs_Victoria Beckham'
		dist2 ='Elton John_vs_Paul McCartney'

		cos_distances = calculate_cos_distance(name, famous_people, people_info)

		print "\nQ3: %s: %s" % (dist1, cos_distances[dist1])
		print "\nQ4: %s: %s" % (dist2, cos_distances[dist2])
		print "\nQ5: closer to 'Elton John is Paul McCartney"

		# cos_distances = calculate_cos_distance('Barack Obama', famous_people, people_info)
		# for dist in cos_distances.keys():
		# 	print "%s: %s"%(dist, cos_distances[dist])

		# 6) Now, you will build two nearest neighbors models:
			# Using word counts as features
			# Using TF-IDF as features
			# set the distance function to cosine similarity

		knn_model_word_count = gp.create_nearest_neighbors_model(people,features=['word_count'],label='name',distance='cosine')
		knn_model_tfidf = gp.create_nearest_neighbors_model(people,features=['tfidf'],label='name',distance='cosine')

		# Whats the most similar article, other than itself
		# Elton John & Victoria Beckham using word count features? & TF-IDF features?
		print "Find the Nearest Neighbor of"
		count_qs = 6
		for name in ['Elton John','Victoria Beckham']:
			query_min_knn_distance(name,people_info,knn_model_word_count,'raw_model',count_qs)
			count_qs+=1
			query_min_knn_distance(name,people_info,knn_model_tfidf,'tfidf_model',count_qs)
			count_qs+=1

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()
