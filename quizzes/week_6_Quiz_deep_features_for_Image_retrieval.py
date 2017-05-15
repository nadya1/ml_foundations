__author__ = 'nadyaK'
__date__ = '04/22/17'

import ml_graphlab_utils as gp
import traceback

def create_train_labels(image_train, categories):
	"""e.g dict={'cat': sframe_train_cat_data, ....}"""
	return {x: image_train[image_train['label'] == x] for x in categories}

def create_labels_nearest_neighbors_model(train_labels, categories):
	"""e.g dict={'cat': knn_model_for_cat, ....}"""
	label_knn_models = {x:gp.create_nearest_neighbors_model(train_labels[x],
														features=['deep_features'],
														label='id') for x in categories}
	return label_knn_models

def get_nearest_distance_id_query(knn_models, label, tet_query):
	""" create a dict from query-info and find min-key distance"""
	current_query = knn_models[label].query(tet_query,verbose=False)
	# print current_query
	cat_distance = gp.convert_sframe_to_simple_dict(current_query,'reference_label','distance')
	return current_query, gp.find_key_min(cat_distance)

def get_label_distances(labels_distances, label_knn_models, image_test_label):
	""" e.g sframe = {'dog-automobile':[distances_img_tets-label_knn_model .......] """
	label_distances = gp.graphlab.SFrame()
	for label in labels_distances:
		label_name = label.split('-')[-1]
		current_query = label_knn_models[label_name].query(image_test_label,k=1,verbose=False)
		label_distances[label] = current_query['distance']
	return label_distances

def is_dog_correct(row):
	dog_dog_vs_dog_auto = row['dog-dog'] < row['dog-automobile']
	dog_dog_vs_dog_bird = row['dog-dog'] < row['dog-bird']
	dog_dog_vs_dog_cat = row['dog-dog'] < row['dog-cat']
	return dog_dog_vs_dog_auto and dog_dog_vs_dog_bird and dog_dog_vs_dog_cat

def main():
	try:
		image_train = gp.load_data('../../data_sets/image_train_data/')
		image_test = gp.load_data('../../data_sets/image_test_data/')

		#1) Computing summary statistics of the data:
		label_col = image_train['label'].sketch_summary()
		# print label_col
		print "\nQ1: least common category: 'bird'"

		#2) Creating category-specific image retrieval models:
		categories = ['automobile', 'cat', 'dog', 'bird']
		train_labels = create_train_labels(image_train, categories)
		# print train_labels.keys()
		knn_models = create_labels_nearest_neighbors_model(train_labels, categories)

		cat_test_query = image_test[0:1]
		# cat_test_query['image'].show() #using ipython it shows the image in browser
		cat_query, cat_distance = get_nearest_distance_id_query(knn_models, 'cat', cat_test_query)
		print "\nQ2: nearest 'cat' labeled image id: %s"%cat_distance
		# # train_labels['cat'][train_labels['cat']['id'] == 16289]['image'].show()

		dog_query, dog_distance = get_nearest_distance_id_query(knn_models, 'dog', cat_test_query)
		print "\nQ3: nearest 'dog' labeled image id: %s"%dog_distance
		# # train_labels['dog'][train_labels['dog']['id'] == 16976]['image'].show()

		#3) A simple example of nearest-neighbors classification:
		#he mean distance between this image and its nearest neighbors in training data?
		print "\nQ4: 'cat' neighbors mean-distance: %s" % cat_query['distance'].mean()
		print "\nQ5: 'dog' neighbors mean-distance: %s" % dog_query['distance'].mean()
		print "\nQ6: in average 1st img in test data is closer to nearest neighbors in cat data"

		#4. Computing nearest neighbors accuracy using SFrame operations:
		test_labels = create_train_labels(image_test, categories)
		image_test_dog = test_labels['dog']
		labels_dog_distances = ['dog-automobile','dog-cat','dog-dog','dog-bird']

		dog_distances = get_label_distances(labels_dog_distances, knn_models, image_test_dog)
		# print 'Dog-distances: \n', dog_distances
		correct_dog_predictions = dog_distances.apply(is_dog_correct)
		# correct_dog_predictions.sketch_summary()
		# print correct_dog_predictions
		accuracy_1knn_dof = (correct_dog_predictions.sum()/float(len(image_test_dog))) * 100
		print "\nQ7: accuracy of 1-knn classifying 'dog' img in test set: %% %.2f"% accuracy_1knn_dof

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()
