import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon
from matplotlib import colors
import seaborn as sns
import pandas as pd
import string
import re
import random
import time

from mlxtend.frequent_patterns import fpgrowth # , apriori, association_rules
from collections import defaultdict
from nltk.corpus import stopwords as sw
from nltk.stem.snowball import ItalianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split, cross_val_predict

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#
# Class for solving the assignment
#
class ExamAssignment:
	
	# Constructor
	def __init__(self):
		return
	
	def sentimentAnalysis(self, dump_results_to_csv=True):

		# column_names = ['text', 'class']
		training_dataset = self.load_dev_dataset()
		evaluation_dataset = self.load_eva_dataset()

		#	# 	----------------------------
		#	# 	DATA EXPLORATION
		#	#
		#	#

		X_train = training_dataset['text']
		y_train = training_dataset['class']
		dataset_size = len(X_train)

		X_train_pos = training_dataset.loc[training_dataset['class'] == 'pos', 'text']
		X_train_neg = training_dataset.loc[training_dataset['class'] == 'neg', 'text']
		
		X_eval = evaluation_dataset['text']
		evaluation_dataset_size = len(X_eval)

		# TRAINING DATASET SIZE: 28754
		# # POS SIZE: 19532
		# # NEG SIZE: 9222
		# EVALUATION DATASET: 12323

		# print(X_train)
		# print(y_train)

		# SHOW HIST ON LENGHT OF DOCUMENTS -----
		# self.show_hist_on_length(X_train, X_train_neg, X_train_pos)
		# --------------------------------------

		# BOXPLOT ON REVIEW LENGTH ------
		# self.show_boxplots_on_length(X_train, X_train_neg, X_train_pos)
		# -------------------------------

		# TOP N WORDS separately for each class ------
		# self.show_top_n_words(X_train, X_train_neg, X_train_pos)
		# --------------------------------------------

		# FREQUENT ITEMSETS ------
		# self.show_frequent_itemsets(X_train)
		# ------------------------

		# REVIEWS AT RANDOM ------
		# self.show_reviews_at_random(X_train)
		# ------------------------


		#	# 	----------------------------
		#	# 	DATA PREPROCESSING, ALGORITHM CHOICE, TUNING AND VALIDATION
		#	#
		#	#

		# GRID SEARCH --------------
		# self.perform_grid_search(X_train, y_train)
		# sys.exit()
		# --------------------------

		myTokenizer = MyTokenizer(no_numbers=True,
								  min_length=1,
								  stemmer=True,
								  clean_emoji=True,
								  stop_words_bool=True,
								  whitelist_stop_words=True)

		vectorizer = TfidfVectorizer(input='content',
							  encoding='utf-8',
							  decode_error='strict',
							  strip_accents=None,
							  lowercase=True,
							  preprocessor=None,
							  tokenizer=myTokenizer,
							  stop_words=None,
							  ngram_range=(1, 2),
							  max_df=0.5,
							  min_df=1,
							  max_features=None)

		# vectorizer.fit(X_train)
		# X_train_tfidf = vectorizer.transform(X_train)

		# vocabulary = vectorizer.vocabulary_ # 53 386 (max_df=0.5, min_df=1, unigrams)
		# words_left_out = vectorizer.stop_words_

		# CLASSIFIERS --------------------------------
		# gnb = GaussianNB() # goes to 0.88
		# mnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None) # goes to 0.92
		# cmb = ComplementNB(alpha=1.0) # this is supposed to work better than multinomial for unbalanced classes
		svc = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.00000001, C=3.0) # goes to 0.960
		# svcgeneral = SVC(kernel='rbf', degree=3, gamma='scale', tol=0.000001, C=2.0)
		# sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000)
		# rfc = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=4, max_features='auto', min_impurity_decrease=0.0001)
		# knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
		# ---------------------------------------------

		# DIMENSIONALITY REDUCTION SEEMS TO NOT HELP ---
		# pca = PCA(n_components=10000)
		# lsa = TruncatedSVD(n_components=400, algorithm='randomized', n_iter=5) #without pca is better (here goes to 0.95)
		# ch2_selection = SelectKBest(score_func=chi2, k=10000) # the lower the worse!
		# normalizer = Normalizer(copy=True)
		# ----------------------------------------------

		# SCALING --------------------------------------
		# scaler = StandardScaler() # does not work with sparse
		# scaler = MaxAbsScaler() # WORKS, but gives worse results
		# scaler = MinMaxScaler() # does not work with sparse
		# scaler = RobustScaler() # does not work with sparse
		# ----------------------------------------------

		clf_pipe = make_pipeline(vectorizer, svc)

		# self.show_confusion_matrix(X_train, y_train, clf_pipe)
		# sys.exit()

		# CROSS-VALIDATION SCORE ---
		f1_weighted = cross_val_score(clf_pipe, X_train, y_train, cv=10, scoring='f1_weighted')
		print(f"Score f1_weighted: {f1_weighted}")
		print(f"Mean f1_weighted: {f1_weighted.mean()}")
		# --------------------------

		if dump_results_to_csv:

			clf_pipe.fit(X_train, y_train)

			# feature_importances = rfc.feature_importances_
			# vocabulary = vectorizer.vocabulary_
			# self.print_top_feature_importances(feature_importances, vocabulary)
			# sys.exit()

			# print(vectorizer.vocabulary_)
			print('vocabulary length: ', len(vectorizer.vocabulary_))

			y_test = clf_pipe.predict(X_eval)

			self.dump_to_csv(y_test)

		return

	def show_confusion_matrix(self, X_train, y_train, clf_pipe):
	
		y_train_pred = cross_val_predict(clf_pipe, X_train, y_train, cv=10)

		conf_mat = confusion_matrix(y_train, y_train_pred)

		print('Confusion matrix: ', conf_mat)

		fig, ax = plt.subplots(nrows=1, ncols=1)
		# sns.set()
		# akws = {"ha": 'center',"va": 'center'}
		# akws = {"ha": 'left',"va": 'center'}
		# sns.heatmap(conf_mat, cmap='Wistia',
		# 			fmt='d',
		# 			xticklabels=['Neg', 'Pos'],
		# 			yticklabels=['Neg', 'Pos'],
		# 			annot=True,
		# 			annot_kws=akws,
		# 			ax=ax)
		# plt.yticks(rotation=180, ha='right', rotation_mode='anchor')

		im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Wistia)
		# ax.figure.colorbar(im, ax=ax)

		ax.set(yticks=[-0.5, 1.5], 
		       xticks=[0, 1], 
		       yticklabels=['Neg', 'Pos'], 
		       xticklabels=['Neg', 'Pos'])
		ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))

		for i in range(2):
		    for j in range(2):
		        text = ax.text(j, i, conf_mat[i, j], ha="center", va="center", color="#00000090")
		
		plt.show()
		return

	def show_hist_on_length(self, X_train, X_train_neg, X_train_pos):

		X_train_lengths = [(len(doc), i) for i, doc in enumerate(X_train)]
		# X_train_lengths_sorted = sorted(X_train_lengths, key=lambda x: x[0])
		X_train_lengths_array = np.array([tup[0] for tup in X_train_lengths])

		# X_train_lengths_pos = [(len(doc), i) for i, doc in enumerate(X_train_pos)]
		# X_train_lengths_array_pos = np.array([tup[0] for tup in X_train_lengths_pos])

		# X_train_lengths_neg = [(len(doc), i) for i, doc in enumerate(X_train_neg)]
		# X_train_lengths_array_neg = np.array([tup[0] for tup in X_train_lengths_neg])

		fig, ax = plt.subplots(nrows=1, ncols=1)
		N, bins, patches = ax.hist(X_train_lengths_array, density=True, bins=50)
		fracs = N/N.max()

		norm = colors.Normalize(fracs.min(), fracs.max())

		for thisfrac, thispatch in zip(fracs, patches):
		    color = plt.cm.Wistia(norm(thisfrac)) #Blues
		    thispatch.set_facecolor(color)

		# N, bins, patches = ax.hist(X_train_lengths_array_neg, density=True, bins=50, alpha=0.75, label='Negative reviews')
		# fracs = N/N.max()
		# norm = colors.Normalize(fracs.min(), fracs.max())

		# for thisfrac, thispatch in zip(fracs, patches):
		#     color = plt.cm.cool(norm(thisfrac)) #Blues
		#     thispatch.set_facecolor(color)

		
		ax.tick_params(axis='both', which='major', labelsize=12)
		# fig.suptitle('LENGTH OF EACH DOCUMENT', fontsize=14)
		ax.set_title('REVIEW LENGTH', alpha=0.85, fontsize=18, weight='normal', pad=25.0)
		ax.set_xlabel('Number of characters contained', fontsize=16, alpha=0.85, weight='normal', labelpad=18.0)

		ax.tick_params(axis='both', which='major', labelsize=14, pad=8.0)
		# print(f"Statistics:\nMean: {X_train_lengths_array.mean()}\nSTD: {X_train_lengths_array.std()}\nMedian: {np.median(X_train_lengths_array)}")

		# ax.legend(fontsize='x-large')
		plt.grid(alpha=0.25)
		plt.show()

		return

	def show_boxplots_on_length(self, X_train, X_train_neg, X_train_pos):

		X_train_lengths = [(len(doc), i) for i, doc in enumerate(X_train)]
		X_train_lengths_array = np.array([tup[0] for tup in X_train_lengths])

		X_train_lengths_pos = [(len(doc), i) for i, doc in enumerate(X_train_pos)]
		X_train_lengths_array_pos = np.array([tup[0] for tup in X_train_lengths_pos])

		X_train_lengths_neg = [(len(doc), i) for i, doc in enumerate(X_train_neg)]
		X_train_lengths_array_neg = np.array([tup[0] for tup in X_train_lengths_neg])

		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.set_title('REVIEW LENGTH BY CLASS', alpha=0.85, fontsize=18, weight='normal', pad=25.0)
		bp = ax.boxplot([X_train_lengths_array,X_train_lengths_array_pos, X_train_lengths_array_neg],
				   labels=['Overall', 'Positive', 'Negative'],
				   showfliers=False,
				   showmeans=False)

		plt.setp(bp['boxes'][0], color='#00000095', linewidth=2)
		plt.setp(bp['boxes'][1], color='#00800095', linewidth=2)
		plt.setp(bp['boxes'][2], color='#80000095', linewidth=2)

		# MEDIANS
		plt.setp(bp['medians'][0], color='#00000095', linewidth=2)
		plt.setp(bp['medians'][1], color='#00800095', linewidth=2)
		plt.setp(bp['medians'][2], color='#80000095', linewidth=2)

		# HORIZONTAL CAPS
		plt.setp(bp['caps'][0], color='#00000095', linewidth=2)
		plt.setp(bp['caps'][1], color='#00000095', linewidth=2)

		plt.setp(bp['caps'][2], color='#00800095', linewidth=2)
		plt.setp(bp['caps'][3], color='#00800095', linewidth=2)

		plt.setp(bp['caps'][4], color='#80000095', linewidth=2)
		plt.setp(bp['caps'][5], color='#80000095', linewidth=2)

		# VERTICAL LINES
		plt.setp(bp['whiskers'][0], color='#00000095', linewidth=2)
		plt.setp(bp['whiskers'][1], color='#00000095', linewidth=2)

		plt.setp(bp['whiskers'][2], color='#00800095', linewidth=2)
		plt.setp(bp['whiskers'][3], color='#00800095', linewidth=2)
		
		plt.setp(bp['whiskers'][4], color='#80000095', linewidth=2)
		plt.setp(bp['whiskers'][5], color='#80000095', linewidth=2)

		num_boxes = 3
		box_colors = ['#00000050', '#00800050', '#80000050']
		medians = np.empty(num_boxes)

		for i in range(num_boxes):
			box = bp['boxes'][i]
			boxX = []
			boxY = []
			for j in range(5):
				boxX.append(box.get_xdata()[j])
				boxY.append(box.get_ydata()[j])

			box_coords = np.column_stack([boxX, boxY])
			ax.add_patch(Polygon(box_coords, facecolor=box_colors[i]))

		plt.grid(alpha=0.25)
		ax.tick_params(axis='both', which='major', labelsize=18, pad=8.0)
		plt.show()

		return

	def get_top_n_words(self, corpus, n=10):
		myTokenizer = MyTokenizer(no_numbers=True,
								  min_length=1,
								  stemmer=False,
								  stop_words_bool=True)

		countVectorizer = CountVectorizer(encoding='utf-8',
										decode_error='strict',
										strip_accents=None,
										lowercase=True,
										preprocessor=None,
										binary=True,
										max_df=1.0,
										stop_words=None,
										ngram_range=(1,1),
										tokenizer=myTokenizer)

		vec = countVectorizer.fit(corpus)
		bag_of_words = vec.transform(corpus)

		sum_words = bag_of_words.sum(axis=0) 

		words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
		words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
		return words_freq[:n]

	def get_counts_of_given_words(self, corpus, top_words):

		myTokenizer = MyTokenizer(no_numbers=True,
								  min_length=1,
								  stemmer=False,
								  clean_emoji=True,
								  whitelist_stop_words=True,
								  stop_words_bool=True)

		countVectorizer = CountVectorizer(encoding='utf-8',
										decode_error='strict',
										strip_accents=None,
										lowercase=True,
										stop_words=None,
										preprocessor=None,
										binary=True,
										max_df=1.0,
										ngram_range=(1,1),
										tokenizer=myTokenizer)

		vec = countVectorizer.fit(corpus)
		bag_of_words = vec.transform(corpus)

		sum_words = bag_of_words.sum(axis=0)

		top_words_words = list(top_words['word'])
		# top_words_words = list(top_words) # nel caso di all_emojis

		words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if word in top_words_words]
		new_words_freq = []

		for word in top_words_words:
			i = -1
			for j, word_freq in enumerate(words_freq):
				if word_freq[0] == word:
					i = j
					break
			if i == -1:
				new_words_freq.append( tuple([word, 0]) )
				continue

			new_words_freq.append(words_freq[i])

		return new_words_freq

	def show_top_n_words(self, X_train, X_train_neg, X_train_pos):

		top_words = pd.DataFrame(self.get_top_n_words(X_train, 30), columns=['word', 'count'])

		# # pos_emoji = ['😀','💪', '😎','👌','😁', '😃', '😄','😊','😋','😍','😻','🤗','👏🏻','😘','🎉','💗','🔝','👍','😉']
		# # neg_emoji = ['👎', '😒','😖','😠','😡','😤','😨', '😱','😳','😬','😞','🤐','😕','😢']
		# # neutral_emoji = ['😔','😴','😂','😅','😆','😓']
		# pos_emoji = ['👍','😀','💪', '😎','👌','😁','😃','😄','😊','😋','😍','😻','🤗','👏🏻','😘','🎉','💗','🔝','😉']
		# neg_emoji = ['👎','😒','😖','😠','😡','😤','😨','😱','😳','😬','😞','🤐','😕','😢']
		# neutral_emoji = ['😔','😴','😂','😅','😆','😓']
		# # all_emojis = list(set(pos_emoji) | set(neg_emoji))
		# # all_emojis = ['positiveemoji', 'negativeemoji']
		# all_emojis_pos = ['positiveemoji_'+str(i) for i, emoji in enumerate(pos_emoji)]
		# all_emojis_neg = ['negativeemoji_'+str(i) for i, emoji in enumerate(neg_emoji)]
		# all_emojis_neutral = ['neutralemoji_'+str(i) for i, emoji in enumerate(neutral_emoji)]
		# all_emojis = all_emojis_pos + all_emojis_neg + all_emojis_neutral
		# top_words_pos = pd.DataFrame(self.get_counts_of_given_words(X_train_pos, all_emojis), columns=['word', 'count'])
		# top_words_neg = pd.DataFrame(self.get_counts_of_given_words(X_train_neg, all_emojis), columns=['word', 'count'])

		top_words_pos = pd.DataFrame(self.get_counts_of_given_words(X_train_pos, top_words), columns=['word', 'count'])
		top_words_neg = pd.DataFrame(self.get_counts_of_given_words(X_train_neg, top_words), columns=['word', 'count'])

		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.set_title('TOP 30 UNIGRAMS', fontsize=18, pad=25.0, alpha=0.85, weight='normal')
		
		indices = np.arange(len(top_words['count']))
		# indices = np.arange(len(all_emojis))
		# width = np.min(np.diff(indices))/10.
		width = 0.25
		
		ax.bar(indices-(width/2), (top_words_pos['count']/len(X_train_pos)), width, color='#00800050', edgecolor='#00800095', linewidth=0.5, label='Positive class')
		ax.bar(indices+(width/2), (top_words_neg['count']/len(X_train_neg)), width, color='#80000050', edgecolor='#80000095', linewidth=0.5, label='Negative class')

		ax.set_xticklabels(top_words['word'])
		# ax.set_xticklabels(all_emojis)
		ax.set_xticks(indices)
		ax.legend()
		plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
		ax.set_ylabel('Document frequency', alpha=0.85, weight='normal', fontsize=14, labelpad=18.0)
		plt.grid(alpha=0.2)
		
		plt.show()
		return

	def show_frequent_itemsets(self, X_train):

		X_train_tokens, unique_words = self.myTokenizer(X_train)
		filtered_unique_words = [word for word in unique_words if len(word) > 2]
		print('All words: ', len(unique_words)) # 53545
		print('All words longer than 2 char: ', len(filtered_unique_words)) # 53006

		frequent_itemsets = self.getFrequentItemsets(X_train_tokens, filtered_unique_words)

		print(filtered_unique_words)

		return

	def show_reviews_at_random(self, X_train):

		X_train_tokens, unique_words = self.myTokenizer(X_train)

		print(random.sample(X_train_tokens, len(X_train_tokens)))

		return

	def print_top_feature_importances(self, feature_importances, vocabulary):

		indexes = np.argsort(feature_importances)[::-1]

		height = feature_importances[list(indexes[:30])]
		width = 0.5
		xlabels = [(k, v) for k,v in vocabulary.items()]
		xlabels = sorted(xlabels, key=lambda x: x[1])
		actualXLabels = [xlabels[indexes[i]][0] for i in range(len(height))]

		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.set_title('TOP 20 FEATURES', fontsize=18, pad=25.0, alpha=0.85, weight='normal')

		ax.bar(actualXLabels, height, width, color='#FF6F0050', edgecolor='#FF6F0095', linewidth=0.5)

		plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
		ax.set_ylabel('Feature importance', alpha=0.85, weight='normal', fontsize=14, labelpad=18.0)
		plt.grid(alpha=0.2)
		
		plt.show()
		return

	def perform_grid_search(self, X_train, y_train):

		# params = {
		# 	'classifier': ['complementnb', 'multinomialnb', 'linearsvc', 'randomforest', 'knn'],
		# 	'min_length': [1, 2],
		# 	'stop_words_bool': [0, 1],
		# 	'max_df': [0.5],
		# 	'min_df': [2],
		# 	'pca_n_components': [False, 1000]
		# }

		# params = {
		# 	'classifier': ['linearsvc'],
		# 	'AAAstemmer': [False, True],
		# 	'ABBdual': [True, False],
		# 	'min_length': [1, 2],
		# 	'stop_words_bool': [0, 1, 2],
		# 	'max_df': [0.3, 0.5, 0.6],
		# 	'min_df': [1, 2, 4],
		# 	'ngram_range': [(1,1), (1,2), (1,3)],
		# 	'C': [0.1, 0.5, 1.0, 50.0, 100.0],
		# 	'tol': [0.0001, 0.00001],
		# 	'loss': ['squared_hinge', 'hinge'],
		# 	'ABCpenalty': ['l2', 'l1']
		# }

		params = {
			'classifier': ['linearsvc'],
			'AAAstemmer': [True],
			'ABBdual': [True],
			'min_length': [1],
			'stop_words_bool': [2],
			'max_df': [0.5, 0.8],
			'min_df': [1, 2],
			'ngram_range': [(1,2)],
			'C': [2.0, 2.5, 3.0],
			'tol': [0.000001],
			'loss': ['squared_hinge', 'hinge'],
			'ABCpenalty': ['l2']
		}

		# params = {
		# 	'classifier': ['complementnb', 'decisiontree', 'linearsvc', 'randomforest', 'knn'],
		# 	'min_length': [1],
		# 	'stop_words_bool': [1],
		# 	'max_df': [0.5],
		# 	'min_df': [2],
		# 	'scaler': ['none', 'standard', 'minmax', 'maxabs', 'robust']
		# }

		# params = {
		# 	'classifier': ['linearsvc'],
		# 	'min_length': [1, 2, 3]
		# }

		with open('___gridsearch_da_definire.csv', 'w', encoding='utf-8') as file:
			print(f"CLASSIFIER,STEMMER,DUAL,STOP_WORDS_FLAG,MIN_LENGTH,NGRAM_1,NGRAM_2,MAX_DF,MIN_DF,C,LOSS,TOL,PENALTY,F1_WEIGHTED", file=file)
			# print(f"CLASSIFIER,SCALER,F1_WEIGHTED", file=file)


		bestScore = 0
		bestConfig = None
		
		for configuration in ParameterGrid(params):			
			print(f"--------------")
			print(f"Current config: {configuration}")
			start = time.time()
			score = self.score_with_this_config(X=X_train, y=y_train, **configuration)
			end = time.time()
			print('TIME ELAPSED: ', (end - start))
			print(f"--------------")

			if score > bestScore:
				bestScore = score
				bestConfig = configuration

			print(f"((((((((((((( BEST SCORE SO FAR: {bestScore} ))))))))))))))")

		print(f"----------------------------")
		print(f"----------------------------")
		print(f"----------------------------")
		print(f"----------------------------")
		print(f"----------------------------")
		print(f"----------------------------")
		print(f"BEST CONFIG FOUND: {bestConfig}")
		print(f"WITH BEST SCORE: {bestScore}")

		return

	def score_with_this_config(self,
							   X=None,
							   y=None,
							   min_length=1,
							   AAAstemmer=False,
							   ABBdual=True,
							   stop_words_bool=0, # 0, 1, 2 -> (False, True with no whitelist, True with whitelist)
							   ngram_range=(1, 1),
							   max_df=0.5,
							   min_df=2,
							   pca_n_components=False,
							   classifier='linearsvc',
							   C=1.0,
							   loss='squared_hinge',
							   tol=0.0001,
							   ABCpenalty='l2',
							   scaler='none',
							   max_features=None):

		stop_words = None
		if stop_words_bool != 0:
			if stop_words_bool == 2:
				sw = StopWords(whitelist=True)
			else:
				sw = StopWords()

			stop_words = sw.getStopWords()

		myTokenizer = MyTokenizer(no_numbers=True,
								  min_length=min_length,
								  stemmer=AAAstemmer)

		vectorizer = TfidfVectorizer(strip_accents=None,
							  		 lowercase=True,
							 		 tokenizer=myTokenizer,
							  		 ngram_range=ngram_range,
							  		 stop_words=stop_words,
							  		 max_df=max_df,
							 		 min_df=min_df,
							 		 max_features=max_features)

		
		if classifier == 'gaussiannb':
			clf = GaussianNB() # goes to 0.88
		elif classifier == 'multinomialnb':
			clf = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None) # goes to 0.92
		elif classifier == 'complementnb':
			clf = ComplementNB(alpha=1.0)
		elif classifier == 'linearsvc':
			clf = LinearSVC(penalty=ABCpenalty, loss=loss, dual=ABBdual, tol=tol, C=C) # goes to 0.960
		elif classifier == 'randomforest':
			clf = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, max_features='auto', min_impurity_decrease=0.0)
		elif classifier == 'decisiontree':
			clf = DecisionTreeClassifier(max_depth=None, splitter='best', min_impurity_decrease=0.005, min_samples_split=10)
		elif classifier == 'sgd':
			clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000)
		elif classifier == 'knn':
			clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
		else:
			print(f"FATAL ERROR -----")
			sys.exit()

		if pca_n_components != False and (classifier == 'gaussiannb' or classifier == 'multinomialnb' or classifier == 'complementnb'):
			print(f"SKIPPED")
			return

		if scaler == 'none':
			scaleTransformer = VoidTransformer()
		elif scaler == 'standard':
			scaleTransformer = StandardScaler()
		elif scaler == 'minmax':
			scaleTransformer = MinMaxScaler()
		elif scaler == 'maxabs':
			scaleTransformer = MaxAbsScaler()
		elif scaler == 'robust':
			scaleTransformer = RobustScaler()
		else:
			print(f"FATAL ERROR ON SCALER NAME")
			sys.exit()

		if pca_n_components != False:
			lsa = TruncatedSVD(n_components=pca_n_components, algorithm='randomized', n_iter=5)
			normalizer = Normalizer(copy=True)
			if classifier == 'gaussiannb':
				clf_pipe = make_pipeline(vectorizer, lsa, normalizer, DenseTransformer(), clf)
			else:
				clf_pipe = make_pipeline(vectorizer, lsa, normalizer, clf)
		else:
			if classifier == 'gaussiannb':
				clf_pipe = make_pipeline(vectorizer, scaleTransformer, DenseTransformer(), clf)
			else:
				clf_pipe = make_pipeline(vectorizer, scaleTransformer, clf)

		f1_weighted = cross_val_score(clf_pipe, X, y, cv=10, scoring='f1_weighted')
		score_mean = f1_weighted.mean()

		print('F1_WEIGHTED: ', score_mean)

		with open('___gridsearch_da_definire.csv', 'a', encoding='utf-8') as file:
			print(f"{classifier},{AAAstemmer},{ABBdual},{stop_words_bool},{min_length},{ngram_range[0]},{ngram_range[1]},{max_df},{min_df},{C},{loss},{tol},{ABCpenalty},{score_mean}", file=file)
			# print(f"{classifier},{scaler},{score_mean}", file=file)
				
		return score_mean


	#
	# REQUIRES REALLY HIGH MEMORY AND TIME, BETTER USE COUNTVECTORIZER WITH ITS OWN PARAMETERS
	#
	def getFrequentItemsets(self, corpus_of_tokens, unique_words, minsup=0.2):

		items = list(unique_words)

		n_transactions = len(corpus_of_tokens)
		n_items = len(items)
		# mapping_items = {item:i for i, item in enumerate(items)}

		mapping_items = defaultdict(lambda: -1)
		for i, item in enumerate(items):
			mapping_items[item] = i

		matrix = np.zeros( (n_transactions, n_items))

		for i, row in enumerate(corpus_of_tokens):
			currUsed = set()
			for token in row:
				if mapping_items[token] != -1:
					if token not in currUsed:
						matrix[i][mapping_items[token]] = 1
						currUsed.add(token)

		df = pd.DataFrame(data=matrix, columns=items)
		fi = fpgrowth(df, minsup, use_colnames=True)

		fi.to_csv('frequent_itemsets_above2_above20sup.csv', sep=',')
		# print(fi.tolist())

		return

	#
	# DEPRECATED FUNCTION -> USE MyTokenizer() CLASS INSTEAD
	#
	def myTokenizer(self, corpus, no_numbers=True, min_length=1, stop_words_bool=False, stemmer=False):
		tokens = []
		unique_words = set()
		replace = ['>', '<', '-', '|', '\\', '/', '^', '\n', '”', '“', '"', '’', '...']

		ita_stemmer = ItalianStemmer()

		for doc in corpus:
			if no_numbers:
				doc = re.sub(r'\d+', '', doc)

			for punct in string.punctuation:
				doc = doc.replace(punct, " ")

			for specialChar in replace:
				doc = doc.replace(specialChar, ' ')

			split_doc = [ token.lower().strip() for token in doc.split(" ") if token ]

			split_doc = [word for word in split_doc if len(word) > min_length]

			if stemmer:
				split_doc = [ita_stemmer.stem(word) for word in split_doc]

			unique_words.update(set(split_doc))
			tokens.append(split_doc)
		return tokens, unique_words

	def load_eva_dataset(self):

		file_name = 'evaluation.csv'

		evaluation_dataset = pd.read_csv(file_name, skiprows=0, sep=',', encoding='utf-8')

		return evaluation_dataset

	def load_dev_dataset(self):

		file_name = 'development.csv'

		training_dataset = pd.read_csv(file_name, skiprows=0, sep=',', encoding='utf-8')				

		return training_dataset

	def dump_to_csv(self, y_test):
		df_len = len(y_test)

		df = pd.DataFrame({'Id': np.arange(0, df_len), 'Predicted': y_test})

		print(df)

		df.to_csv('output_labels_eval_set.csv', encoding='utf-8', index=False)

		return

#
# My tokenizer class for the TfidfVectorizer
#
class MyTokenizer:
	
	# Constructor
	def __init__(self, no_numbers=True, min_length=1, clean_emoji=True, stop_words_bool=True, whitelist_stop_words=True, stemmer=False):

		self.no_numbers = no_numbers
		self.min_length = min_length
		self.stemmer = stemmer
		self.ita_stemmer = ItalianStemmer()

		self.replace = ['#','>','_','<', '-', '|', '\\', '/', '^', '\n', '”', '“', '"', '’', '‘', '€','´','.', '…']

		self.emoji = None
		self.clean_emoji = clean_emoji
		if self.clean_emoji:
			try:
			    # UCS-4
			    self.emoji = re.compile(u'[\U00010000-\U0010ffff]')
			except re.error:
			    # UCS-2
			    self.emoji = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')

			self.pos_emoji = ['👍','😀','💪', '😎','👌','😁', '😃', '😄','😊','😋','😍','😻','🤗','👏🏻','😘','🎉','💗','🔝','😉']
			self.neg_emoji = ['👎', '😒','😖','😠','😡','😤','😨', '😱','😳','😬','😞','🤐', '😕','😢']

		self.stop_words_bool = stop_words_bool
		if self.stop_words_bool:
			sw = StopWords(whitelist=whitelist_stop_words)
			sw_list = sw.getStopWords()
			
			stop_words_dict = defaultdict(lambda: -1)
			for i, word in enumerate(sw_list):
				stop_words_dict[word] = 1

			self.stop_words_dict = stop_words_dict

		return

	def __call__(self, doc):

		doc = re.sub(r'[hHtTpP]+[sS]?:[A-Za-z0-9-#_./]+', ' ', doc)

		if self.no_numbers:
			doc = re.sub(r'\d+', ' ', doc)

		for punct in string.punctuation:
			doc = doc.replace(punct, " ")

		for specialChar in self.replace:
			doc = doc.replace(specialChar, ' ')

		if self.clean_emoji:

			for specialEmoji in self.pos_emoji:
				doc = doc.replace(specialEmoji, ' positiveemoji ')

			for specialEmoji in self.neg_emoji:
				doc = doc.replace(specialEmoji, ' negativeemoji ')
			
			doc = self.emoji.sub(u' ', doc)

		split_doc = [ token.lower().strip() for token in doc.split(" ") if token ]

		if self.stop_words_bool:
			split_doc = [word for word in split_doc if len(word) > self.min_length and len(word) < 16 and self.stop_words_dict[word] != 1]
		else:
			split_doc = [word for word in split_doc if len(word) > self.min_length and len(word) < 16]

		if self.stemmer:
			split_doc = [self.ita_stemmer.stem(word) for word in split_doc]

		return split_doc

#
# My Stop words class
#
class StopWords:
	
	# Constructor
	def __init__(self, whitelist=False):

		my_stop_words = set(['http', 'www', 'per', 'che', 'con', 'era', 'del', 'della', 'tutto', 'come', 'ecc', 'etc', 'il','li','un','una'])

		sw_italian_nltk = sw.words('italian')
		nltk_stop_words = set(sw_italian_nltk)

		self.stop_words = my_stop_words | nltk_stop_words

		if whitelist:
			stop_words_whitelist = set(['non', 'ma', 'stato', 'personale', 'molto', 'solo', 'quanto'])
			self.stop_words = self.stop_words - stop_words_whitelist
		return

	def getStopWords(self):
		return self.stop_words


class DenseTransformer():

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

class VoidTransformer():
	def fit(self, X, y=None, **fit_params):
		return self

	def transform(self, X, y=None, **fit_params):
		return X


def main():
	# callable through command line "python assignment_solution.py"
	exam = ExamAssignment()

	exam.sentimentAnalysis()

if __name__ == '__main__':
	main()
