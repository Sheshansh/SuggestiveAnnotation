import numpy as np
from scipy import spatial
import operator

predictions, encodings, dice_scores = np.array([None]*4), np.array([None]*4), np.array([None]*4)
for i in range(4):
	predictions[i], encodings[i], dice_scores[i] = np.load('train_eval_data_'+str(i)+'.npy')

keys = dice_scores[0].keys()
K, k = 20, 10

scores = {}
for key in keys:
	total_array = np.array([
			predictions[0][key],
			predictions[1][key],
			predictions[2][key],
			predictions[3][key]
		])
	scores[key] = np.sum(np.var(total_array,axis=0))
# scores list would have images tuples in decreasing order of variance
scores_list = sorted(scores.items(), key=lambda x: -x[1])
options_available = [scores_list[i][0] for i in range(K)]

def similarity(encodings, key1, key2): # cosine similarity
	similarities = [None]*4
	for i in range(4):
		similarities[i] = 1 - spatial.distance.cosine(np.mean(encodings[i][key1],axis=3).flatten(),np.mean(encodings[i][key2],axis=3).flatten())
	return np.mean(similarities)

def unit_F(similarities,key,images_selected):
	max_sim = float('-inf')
	for image_selected in images_selected:
		max_sim = max(max_sim,similarities[(key,image_selected)])
	return max_sim

def calc_F(similarities,keys,images_selected):
	F = 0
	for key in keys:
		F += unit_F(similarities,key,images_selected)
	return F

similarities = {}
for key1 in keys:
	for key2 in keys:
		similarities[(key1,key2)] = similarity(encodings,key1,key2)

# select first 20 images with highest uncertainity and then select
# the 10 representative out of them
images_selected = []
while len(images_selected) < k:
	F_options = {}
	for option in options_available:
		F_options[option] = calc_F(similarities,keys,images_selected+[option])
	best_option = max(F_options.iteritems(), key=operator.itemgetter(1))[0]
	options_available.remove(best_option)
	images_selected.append(best_option)
print(images_selected)