__author__ = 'nadyaK'
__date__ = '04/05/17'

import ml_graphlab_utils as gp
import traceback

def evaluate_house_price_models(info_txts, models, test_data):
	info_txt1, info_txt2 = info_txts
	models1, models2 = models
	evaluate1, evaluate2 = models1.evaluate(test_data), models2.evaluate(test_data)
	print 'Model 1 %s   : %s' % (info_txt1, evaluate1)
	print 'Model 2 %s: %s' % (info_txt2, evaluate2)
	return evaluate1, evaluate2

def predict_house_price_models(info_txt, models, SFrame):
	"""dataset : SFrame | pandas.Dataframe"""
	info_txt1, info_txt2 = info_txt
	models1, models2 = models
	print 'Model 1 %s   : %s' % (info_txt1, models1.predict(SFrame))
	print 'Model 2 %s: %s' % (info_txt2, models2.predict(SFrame))

def find_highest_house_price(sales):
	houses_highest_price = sales[sales['zipcode'] == '98039']
	avg_house = houses_highest_price['price'].mean()
	return avg_house

def filter_data(sales):
	# select the houses that have sqft_living higher than 2000 sqft
	# but no larger than 4000 sqft.
	houses_selected_sqft = sales[sales['sqft_living'].apply(lambda x:2000 < x < 4000)]
	return houses_selected_sqft.num_rows()

def build_regression_model(train_data):
	advanced_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode',
						'condition', #conditionofhouse
						'grade',#measureofqualityofconstruction
						'waterfront',#waterfrontproperty
						'view',#typeofview
						'sqft_above',#squarefeetaboveground
						'sqft_basement',#squarefeetinbasement
						'yr_built',#theyearbuilt
						'yr_renovated',#theyearrenovated
						'lat','long',#thelat-longoftheparcel
						'sqft_living15',#averagesq.ft.of15nearestneighbors
						'sqft_lot15',#averagelotsizeof15nearestneighbors
					    ]
	my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
	my_features_model = gp.create_linear_regression(train_data,target='price',features=my_features)
	advanced_features_model = gp.create_linear_regression(train_data,target='price',features=advanced_features)
	models = [my_features_model,advanced_features_model]
	return models

def week2_summary(sales, train_data, test_data):
	#Build a regression model with 1 feature -> 'sqft_living'
	sqft_model = gp.create_linear_regression(train_data,target='price',features=['sqft_living'])

	print 'Price test-mean: %s' % test_data['price'].mean()
	#543054.042563

	print 'Price model evaluate: %s' % sqft_model.evaluate(test_data)
	#{'max_error': 4143550.8825285914, 'rmse': 255191.02870527367}
	# import matplotlib.pyplot as plt
	# plt.plot(test_data['sqft_living'], test_data['price'],'.',
	# 		 test_data['sqft_living'], sqft_model.predict(test_data),'-')
	# plt.show()

	print 'model coefficients: %s\n' % sqft_model.get('coefficients')
	print 'columns name: %s' % sales.column_names()

	# print "sales[my_features] %s\n" % sales[my_features].show()
	# sales[my_features].show()
	# sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

	#******************
	#   CREATE MODEL  *
	#******************
	#Build a regression model with more features
	my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
	print 'my_features: %s' % my_features
	print '\n1) CREATE model:'
	my_features_model = gp.create_linear_regression(train_data,target='price',features=my_features)

	#******************
	#  EVALUATE MODEL *
	#******************
	info_text = ['(1 feature)','(more feature)']
	models = [sqft_model,my_features_model]
	print '\n2) EVALUATE model:'
	evaluate_house_price_models(info_text,models,test_data)

	#******************
	#  PREDICT MODEL  *
	#******************
	#The first house we will use is considered an "average" house in Seattle.
	house1 = sales[sales['id'] == '5309101200']
	print '\n3) PREDICT model:'
	print '\nhouse1:                      %s' % house1['price']
	predict_house_price_models(info_text,models,house1)

	house2 = sales[sales['id'] == '1925069082']
	print '\nhouse2:                      %s' % house2['price']
	predict_house_price_models(info_text,models,house2)

	bill_gates = {'bedrooms':[8],'bathrooms':[25],'sqft_living':[50000],'sqft_lot':[225000],'floors':[4],
		'zipcode':['98039'],'condition':[10],'grade':[10],'waterfront':[1],'view':[4],'sqft_above':[37500],
		'sqft_basement':[12500],'yr_built':[1994],'yr_renovated':[2010],'lat':[47.627606],'long':[-122.242054],
		'sqft_living15':[5000],'sqft_lot15':[40000]}

	#model receive SFrame not dicts
	house_bill_gates = gp.load_data(bill_gates)
	print '\nhouse-Bill-Gates'
	predict_house_price_models(info_text,models,house_bill_gates)


def main():
	try:
		sales = gp.load_data('../../data_sets/home_data.gl/')
		train_data, test_data = gp.split_data(sales, 0.8)

		#week2_summary(sales,train_data,test_data)

		total_houses = sales.num_rows()
		print '\nData -Total (rows): %s' % total_houses

		#1. Selection and summary statistics
		avg_house = find_highest_house_price(sales)
		print '\n1) Highest average house price: $%s' % avg_house

		#2. Filtering Data
		num_houses_high = filter_data(sales)
		print '\n2) Selected Houses (sqft_living):%s' % num_houses_high

		#3. Building a regression model with several more features
		info_text = ['(my features)','(advanced features)']
		models = build_regression_model(train_data)
		print '\n3) Building a regression model (++features)'
		evaluate1, evaluate2 = evaluate_house_price_models(info_text, models , test_data)

		print "\nAnswers:"
		print "\nQ1: %s" %avg_house
		print "\nQ2: %s" % (num_houses_high/float(total_houses))
		print "\nQ3: %s" %(evaluate1['rmse']-evaluate2['rmse'])

	except Exception as details:
		print "Error >> %s" % details
		traceback.print_exc()

if __name__ == "__main__":
	main()