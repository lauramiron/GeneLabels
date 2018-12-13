import pdb, glob, pickle, sys, os, shutil
import numpy as np
from sklearn import svm, ensemble
from sklearn.neighbors import NearestNeighbors
# from shlex import quote
import networkx
import obonet
import subprocess
import time
import argparse
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete.CPD import TabularCPD
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt

RAND_SEED=0
RAND_STATE=np.random.RandomState(seed=RAND_SEED)
RAND_STATE2=np.random.RandomState(seed=RAND_SEED+1)
DATA_DIR =r'data/'
OUTPUT_DIR=r'output/'
RESULTS_DIR=r'results/'
RUNNING_MODEL_DIR = RESULTS_DIR+'1-running_model'
MODEL_CPDS_FILE = RUNNING_MODEL_DIR+'/model_cpds.p'
MODEL_CPDS_DIR = RUNNING_MODEL_DIR+'/model_cpds_states'
MODEL_CPDS_STATES_DIR = RUNNING_MODEL_DIR+'/model_cpds'
TRUE_LABEL_CPDS_FILE = RUNNING_MODEL_DIR+'/true_label_cpds.p'
MICROARRAY_SERIES = ['GSM992', 'GSM1000', 'GSM993', 'GSM994', 'GSM995', 'GSM996', 'GSM998', 'GSM1004', 'GSM1005', 'GSM1006', 'GSM1008', 'GSM1012', 'GSM1015', 'GSM1007', 'GSM1009', 'GSM1013', 'GSM1014', 'GSM1105', 'GSM1100', 'GSM1101', 'GSM1104', 'GSM895', 'GSM1106', 'GSM1107', 'GSM1102', 'GSM1103', 'GSM1111', 'GSM899', 'GSM1041', 'GSM1047', 'GSM1042', 'GSM1043', 'GSM1044', 'GSM1045', 'GSM1046', 'GSM1055', 'GSM1029', 'GSM1030', 'GSM1032', 'GSM1033', 'GSM1034', 'GSM1048', 'GSM1049', 'GSM1050', 'GSM1051', 'GSM1052', 'GSM1053', 'GSM1054', 'GSM1075', 'GSM1076', 'GSM1090', 'GSM1077', 'GSM1078', 'GSM883', 'GSM930', 'GSM929', 'GSM928', 'GSM926', 'GSM925', 'GSM854', 'GSM855', 'GSM856', 'GSM857', 'GSM864', 'GSM865', 'GSM868', 'GSM872', 'GSM1002', 'GSM1003', 'GSM842', 'GSM843', 'GSM844', 'GSM845', 'GSM846', 'GSM847', 'GSM848', 'GSM849', 'GSM850', 'GSM851', 'GSM880', 'GSM881', 'GSM882', 'GSM874', 'GSM875', 'GSM876', 'GSM877', 'GSM878', 'GSM879', 'GSM972', 'GSM1039', 'GSM1040', 'GSM1037', 'GSM938', 'GSM939', 'GSM907', 'GSM990', 'GSM991', 'GSM997', 'GSM999', 'GSM1001', 'GSM971', 'GSM1057', 'GSM1058', 'GSM1059', 'GSM1060', 'GSM1061', 'GSM1063', 'GSM1064', 'GSM961', 'GSM962', 'GSM963', 'GSM964', 'GSM965', 'GSM966', 'GSM967', 'GSM968', 'GSM1019', 'GSM1020', 'GSM1021', 'GSM1022', 'GSM1023', 'GSM934', 'GSM935', 'GSM936', 'GSM1025', 'GSM937', 'GSM1024', 'GSM918', 'GSM919', 'GSM932', 'GSM933', 'GSM980', 'GSM863', 'GSM921', 'GSM920', 'GSM988', 'GSM922', 'GSM989', 'GSM858', 'GSM902', 'GSM931', 'GSM861', 'GSM862', 'GSM923', 'GSM860', 'GSM924', 'GSM859', 'GSM940', 'GSM942', 'GSM910', 'GSM969', 'GSM970', 'GSM973', 'GSM974', 'GSM975', 'GSM976', 'GSM984', 'GSM977', 'GSM903', 'GSM906', 'GSM985']
GO_NODES_LIST_FILE = DATA_DIR+'go_nodes_list.txt'
GO_LIST_FORMATTED = OUTPUT_DIR+'go_nodes_formatted.txt'
GO_DICT_FILE = OUTPUT_DIR+'go_dict.p'
GENES_DICT_FILE = OUTPUT_DIR+'genes_dict.p'
GO_ANNOTATION_FILE = DATA_DIR+'uniprot-GO-annotations.txt'
GO_GENES_LIST_FILE = OUTPUT_DIR+'go_genes_list.txt'
GO_LABEL_ARR_FILE = OUTPUT_DIR+'go_label_arr.np'
MA_RAW_DICT_FILE = OUTPUT_DIR+'microarray_raw_dict.p'
PAIR_FULL_DICT_FILE =DATA_DIR+'matrix_index.txt'
PAIR_FULL_DATA_FILE = DATA_DIR+'interaction_matrix.np'
MA_DICT_FILE = OUTPUT_DIR+'microarray_dict.p'
MA_WNULLS_DATA = OUTPUT_DIR+'microarray_with_nulls.np'
MA_NONNULL_DATA = OUTPUT_DIR+'microarray_nonnull.np'
OBODB_FILE = DATA_DIR+'go-basic.obo'
ALT_IDS_DICT_FILE = OUTPUT_DIR+'alt_ids_dict.p'
PAIR_DATA_FILE = OUTPUT_DIR+'pairwise_filtered_data.np'

# GENES_LIST_FILE = 'final_genes_list.txt'
# GENES_LIST = open(GENES_FILE).readlines()

def parse_microarray(filename,data_dict={},num_processed=0):
	titles_dict = {}
	with open(filename) as f:
		for line in f:
			if line[0] == '!' or line[0] == '^' or line[0] == '#':
				continue
			if line.split('\t')[0] == 'ID_REF':
				titles = line.split('\t')
				for value in titles:
					if value == 'Gene ID' or value == 'UniGene ID' or value.startswith('GSM'):
						titles_dict[value] = titles.index(value)
			else:
				features = line.split('\t')
				geneid = features[titles_dict['Gene ID']]
				unigeneid = features[titles_dict['UniGene ID']]
				if (geneid == '' or geneid == None) and unigeneid != '':
					pdb.set_trace()
				if unigeneid != '' and (geneid != unigeneid):
					pdb.set_trace()
				if geneid == '' or geneid == None:
					continue
				data_entry = {} if geneid not in data_dict else data_dict[geneid]
				for col_name, index in titles_dict.items():
					if col_name in ('Gene ID','UniGene ID'): continue
					if col_name not in data_entry:
						data_entry[col_name] = [features[index]]
					else:
						data_entry[col_name].append(features[index])
				data_dict[geneid] = data_entry
				num_processed += 1
	print("processed "+str(num_processed)+" rows")
	return data_dict, num_processed

def GDSFiles_to_Dict():
	data_dict = {}
	num_processed = 0
	for marr_file in glob.glob('./microarray/GDS*.soft'):
		print("Starting "+marr_file)
		data_dict, num_processed = parse_microarray(marr_file,data_dict,num_processed)
	num_genes = len(data_dict.keys())
	print("Found "+str(num_genes)+" distinct genes out of "+str(num_processed)+" samples")
	
	# discard genes with missing (missing, not NULL) features
	filtered_dict = {}
	for gene in data_dict.keys():
		if set(list(data_dict[gene].keys())) == set(MICROARRAY_SERIES):
			filtered_dict[gene] = data_dict[gene]
	print("Discarding "+str(num_genes-len(filtered_dict.keys()))+" genes with missing features, leaving "+str(len(filtered_dict.keys()))+" genes with full MA expression data")
	pickle.dump(filtered_dict, open(MA_RAW_DICT_FILE, "wb"))


def ConstructMicroarrayArray(example_key='851262'):
	data_dict = pickle.load(open(MA_RAW_DICT_FILE,'rb'))

	example_features = list(data_dict[example_key].keys())
	if set(example_features) != set(MICROARRAY_SERIES):
		pdb.set_trace()
	num_features = len(example_features)

	features_to_index = {}
	for i in range(num_features):
		features_to_index[example_features[i]] = i
	pickle.dump(features_to_index,open(MA_DICT_FILE,'wb'))
	print("Writing MA index dict to "+str(MA_DICT_FILE))

	genes_dict = pickle.load(open(GENES_DICT_FILE,'rb'))
	num_genes = len(genes_dict.keys())

	print("creating microarray data array with nulls")
	data_arr = np.zeros(shape=(num_genes,num_features))
	m = 0
	for gene in data_dict:
		if gene in genes_dict.keys():
			features = data_dict[gene]
			for feature in features:
				feature_index = features_to_index[feature]
				feature_values = features[feature]
				non_null_values = [value for value in feature_values if value != 'null']
				if len(non_null_values) == 0:
					data_arr[m,feature_index] = np.NaN
				else:
					data_arr[m,feature_index] = np.average(np.array(non_null_values).astype(np.float))
			m+= 1
	np.savetxt(MA_WNULLS_DATA,data_arr)
	print("Removing nulls with Knn")
	RunKnnOnNulls()

def RunKnnOnNulls():
	print("Running K nearest neighbors...")
	data_arr = np.loadtxt(MA_WNULLS_DATA)
	m,f = data_arr.shape
	null_axes = np.isnan(data_arr).any(axis=1)
	feature_averages = np.nanmean(data_arr,axis=0)
	smoothed_data = np.copy(data_arr)
	for i in range(m):
		for j in range(f):
			if np.isnan(smoothed_data[i,j]):
				smoothed_data[i,j] = feature_averages[j]
	
	# for each feature, make model from all training examples that 
	# are not null in that feature
	models = []
	for j in range(f):
		non_null_idxs = np.where(~np.isnan(data_arr[:,j]))
		non_null_data = smoothed_data[non_null_idxs]
		model = NearestNeighbors(n_neighbors=16).fit(non_null_data)
		models.append((model,non_null_idxs))
	
	# iterate through training examples and replace null values with
	# value from that feature of nearest centroid
	for i in range(m):
		if i%1000 == 0: print("completed "+str(i)+" iterations")
		example = np.copy(data_arr[i])
		nan_indices = []
		for j in range(f):
			if np.isnan(example[j]):
				nan_indices.append(j)
				example[j] = feature_averages[j]
		if np.isnan(example).any():
			pdb.set_trace()
		for j in nan_indices:
			model, model_data_idxs = models[j]
			model_data = smoothed_data[model_data_idxs]
			distances, indices = model.kneighbors(example.reshape(1,-1))
			centroid = np.average(non_null_data[indices.flatten()],axis=0,weights=(1/distances.flatten()))
			try:
				assert(np.isnan(centroid).any()==False)
			except:
				pdb.set_trace()
			data_arr[i,j] = centroid[j]
		assert(np.isnan(data_arr[i]).any()==False)
	assert(np.isnan(data_arr).any()==False)
	np.savetxt(MA_NONNULL_DATA,data_arr)


def MakeGeneAndGoDicts():
	# make dict of index-to-GOTerm, using set 105 terms
	go_dict = {}
	with open(GO_LIST_FORMATTED,'r') as f:
		lines = [l.strip() for l in f.readlines()]
		for i in range(len(lines)):
			go_dict[lines[i]] = i
	pickle.dump(go_dict,open(GO_DICT_FILE, "wb"))
	
	# make dict of index-to-GeneId, using intersection of GO annotations, pairwise data, and microarray data
	genes_dict = {}
	MA_genes = set(pickle.load(open(MA_RAW_DICT_FILE,'rb')).keys())
	PAIR_genes = set(pickle.load(open(PAIR_FULL_DICT_FILE,'rb')).keys())
	os.system("cat "+GO_ANNOTATION_FILE+"| tail -n+2 | awk '{{ print $1 }}' > "+GO_GENES_LIST_FILE)
	GO_genes = set([line.strip() for line in open(GO_GENES_LIST_FILE,'r').readlines()])
	i = 0
	for gene in MA_genes:
		if (gene in GO_genes) and (gene in PAIR_genes):
			genes_dict[gene] = i
			i+=1
	pickle.dump(genes_dict,open(GENES_DICT_FILE,'wb'))
	print("MA_genes: "+str(len(MA_genes))+", GO_genes:"+str(len(GO_genes))+", PAIR_genes:"+str(len(PAIR_genes)))
	print("Found "+str(len(genes_dict.keys()))+" intersecting genes across all sets")


def FixOutOfDataGoIds():
	# reformat hardcoded list of curated GO ids
	lines = open(GO_NODES_LIST_FILE,'r').readlines()
	with open(GO_LIST_FORMATTED,'w') as of:
		for l in lines:
			line = l.strip()
			oline = 'GO:0000000'[0:-len(line)]+line
			of.write(oline+'\n')

	# make dictionary mapping old ids to new ids
	obo_graph = obonet.read_obo(OBODB_FILE)
	alt_ids_dic = {}
	all_nodes = []
	for idx, node in obo_graph.nodes(data=True):
		all_nodes.append(idx)
		if 'alt_id' in node:
			if type(node['alt_id']) == list:
				for alt_id in node['alt_id']:
					alt_ids_dic[alt_id] = idx
			else:
				alt_ids_dic[node['alt_id']] = idx
	pickle.dump(alt_ids_dic,open(ALT_IDS_DICT_FILE,'wb'))

	updated=0
	retired=0
	print("Updating curated GO Nodes list ids....")
	lines = [line.strip() for line in open(GO_LIST_FORMATTED,'r').readlines()]
	newlines = []			# check for created duplicates
	with open(GO_LIST_FORMATTED,'w') as f:
		for l in lines:
			line = l.strip()
			if line not in all_nodes:
				if line in alt_ids_dic:
					newid = alt_ids_dic[line]
					print(line+" ------> "+newid)
					updated+=1
					if newid not in newlines:
						f.write(newid+'\n')
						newlines.append(newid)
				else:
					print("Retired: "+line)
					retired += 1
			else:
				if line not in newlines:
					f.write(line+'\n')
					newlines.append(line)
	kept=len(lines)-updated-retired
	print("Out of "+str(len(lines))+" GO nodes in Hierarchical paper, "+str(kept)+" are kept, "+str(updated)+" are updated, and "+str(retired)+" are retired, leaving "+str(len(newlines))+" after removing duplicates")

def update_ids(oldids):
	obo_graph = obonet.read_obo(OBODB_FILE)
	all_nodes = []
	for idx, node in obo_graph.nodes(data=True):
		all_nodes.append(idx)

	alt_ids_dic = pickle.load(open(ALT_IDS_DICT_FILE,'rb'))
	new_ids = []
	for oldid in oldids:
		if (oldid in all_nodes) and oldid not in new_ids:
			new_ids.append(oldid)
		elif oldid in alt_ids_dic:
			newid = alt_ids_dic[oldid]
			if newid not in new_ids:
				new_ids.append(newid)
	return new_ids

def ConstructGoAnnotationArray():
	genes_dict = pickle.load(open(GENES_DICT_FILE,'rb'))
	go_dict = pickle.load(open(GO_DICT_FILE,'rb'))
	inv_go_dict = {v: k for k, v in go_dict.items()}
	m = len(genes_dict.keys())
	f = len(go_dict.keys())

	# -1 indicates negative example, 0 is neither positive nor negative
	alt_ids_dic = pickle.load(open(ALT_IDS_DICT_FILE,'rb'))
	go_labels = np.zeros((m,f))	
	lines = open(GO_ANNOTATION_FILE,'r').readlines()
	for x in range(1,len(lines)):
		line = lines[x]
		values = line.split('\t')
		geneids = values[0].split(',')
		for geneid in geneids:
			if geneid in genes_dict.keys():
				goids = [v.strip(';') for v in values[2].split(' ')]
				gene_idx = genes_dict[geneid]
				for go_id in goids:
					new_id = go_id if go_id not in alt_ids_dic else alt_ids_dic[go_id]
					if new_id in go_dict:
						go_idx = go_dict[new_id]
						go_labels[gene_idx,go_idx] = 1

	# mark parents as non-negative examples of all children
	obo_graph = obonet.read_obo(OBODB_FILE)
	for i in range(m):
		pos_idxs = np.argwhere(go_labels[i]==1).flatten()
		for j in pos_idxs:
			goid = inv_go_dict[j]
			children = networkx.ancestors(obo_graph,goid)
			for child in children:
				if child in go_dict:
					child_idx = go_dict[child]
					if go_labels[i,child_idx] == 0:
						go_labels[i,child_idx] = -1      # mark as non-negative example
			parents = networkx.descendants(obo_graph,goid)
			for parent in parents:			
				if parent in go_dict:
					parent_idx = go_dict[parent]
					go_labels[i,parent_idx] = 1

	np.savetxt(GO_LABEL_ARR_FILE,go_labels)
	print("Constructed go annotation array with "+str(m)+" examples and "+str(f)+" annotations")


def ConstructPairwiseArray():
	full_pairwise_data = np.loadtxt(PAIR_FULL_DATA_FILE)
	full_pairwise_dict = pickle.load(open(PAIR_FULL_DICT_FILE,'rb'))
	common_genes_dict = pickle.load(open(GENES_DICT_FILE,'rb'))
	num_genes = len(common_genes_dict.keys())
	f = full_pairwise_data.shape[1]
	
	pairwise_data = np.zeros(shape=(num_genes,f))
	for geneid,idx in common_genes_dict.items():
		pairwise_data[idx] = full_pairwise_data[full_pairwise_dict[geneid]]
	np.savetxt(PAIR_DATA_FILE,pairwise_data)


def GetKMaxLabels(k=1):
	# go_dict = pickle.load(open(GO_DICT_FILE,'rb'))
	go_data = np.load(GO_LABEL_ARR_FILE)
	# inv_go_dict = {v: k for k, v in go_dict.items()}
	return -(go_data.sum(axis=0)).argsort()[:k]


def LoadCombinedData():
	pairwise_data = np.loadtxt(PAIR_DATA_FILE)
	ma_data = np.loadtxt(MA_NONNULL_DATA)
	data = np.concatenate((pairwise_data,ma_data),axis=1)
	label_data = np.loadtxt(GO_LABEL_ARR_FILE)
	assert(data.shape[0]==label_data.shape[0])
	go_dict = pickle.load(open(GO_DICT_FILE,'rb'))
	genes_dict = pickle.load(open(GENES_DICT_FILE,'rb'))
	return data, label_data, {v: k for k, v in go_dict.items()}, {v: k for k, v in genes_dict.items()}


def load_label_data():
	label_data = np.loadtxt(GO_LABEL_ARR_FILE)
	assert(label_data.shape[0]==label_data.shape[0])
	go_dict = pickle.load(open(GO_DICT_FILE,'rb'))
	return label_data, go_dict


def _timeModelDir(model_name='lineardefault'):
	timestring = time.strftime("%m-%d-%H.%M")
	dirname = RESULTS_DIR+model_name+'_'+timestring
	os.mkdir(dirname)
	return dirname


def DefaultParametersFullData(kernel='linear',C=1.0,gamma='auto',n_estimators=1,resume=True,startID=0,subtree=False):
	print('-----Running SVM on all GO IDs-----')   
	training_data, training_labels, go_inv_dict, genes_inv_dict = LoadCombinedData()
	# models_dict = {}

	model_dir = RUNNING_MODEL_DIR
	scores_file = os.path.join(model_dir,'scores.txt')
	if (resume==True) and os.path.isdir(model_dir):
		startID = len(glob.glob(model_dir,'model-*.p'))
	else:
		if os.path.isdir(model_dir): shutil.rmtree(model_dir)
		os.mkdir(model_dir)
		open(scores_file,'w+').write("GOID\tscore\tkernel\tC\tgamma\n")
	
	# for i in range(startID,training_labels.shape[1]):
	# 	goid = go_inv_dict[i]
	i=1
	for goid in _subtree_labels():
		print(i,'/',len(_subtree_labels()),' ',goid)
		i+=1
		model_training_data = training_data[np.where(training_labels[:,i]!=-1)]
		model_training_labels = training_labels[np.where(training_labels[:,i]!=-1)][:,i]
		if not (model_training_labels==1).any(): model, score = None, -1
		else:
			model, score = test_svm_model(kernel,model_training_data,model_training_labels,C,gamma,n_estimators)
		# models_dict[goid] = model
		open(scores_file,'a').write(goid+'\t'+str(score)+'\t'+str(kernel)+'\t'+str(gamma)+'\n')
		print(str(i+1)+'/'+str(training_labels.shape[1])+' ID:'+goid+' acc: '+str(score))
		pickle.dump(model,open(os.path.join(model_dir,'model-'+goid.split(':')[1]+'.p'),'wb'))
	# pickle.dump(models_dict,open(os.path.join(model_dir,'all_models.p'),'wb'))
	# os.rename(model_dir,_timeModelDir(model_name='lineardefault'))


def make_test_set(training_examples,training_labels,rstate=RAND_STATE,num_test_samples=None):
	num_test_samples = int(training_examples.shape[0]*0.632) if (num_test_samples == None) else num_test_samples
	test_idxs = rstate.choice(range(0,training_examples.shape[0]),size=num_test_samples,replace=True)
	test_set = np.zeros(shape=(num_test_samples,training_examples.shape[1]))
	label_shape = 1 if (len(training_labels.shape) < 2) else training_labels.shape[1]
	test_labels = np.zeros(shape=(num_test_samples,label_shape))
	for i in range(0,num_test_samples):
		test_set[i] = training_examples[test_idxs[i]]
		test_labels[i] = training_labels[test_idxs[i]]
	return test_set, test_labels, test_idxs

def make_label_set(training_labels,rstate=RAND_STATE,num_test_samples=None):
	num_test_samples = int(training_labels.shape[0]*0.632) if (num_test_samples==None) else num_test_samples
	test_idxs = rstate.choice(range(0,training_labels.shape[0]),size=num_test_samples,replace=True)
	label_shape = 1 if (len(training_labels.shape) < 2) else training_labels.shape[1]
	test_labels = np.zeros(shape=(num_test_samples,label_shape))
	for i in range(0,num_test_samples):
		test_labels[i] = training_labels[test_idxs[i]]
	return test_labels

def test_svm_model(kernel, training_examples, training_labels, C=1.0, gamma='auto',n_estimators=10):
	model = ensemble.BaggingClassifier(svm.SVC(kernel=kernel, gamma=gamma, random_state=RAND_SEED, probability=True), n_estimators=n_estimators,max_samples=0.632)
	model.fit(training_examples, training_labels)
	test_set, test_labels = make_test_set(training_examples,training_labels)
	test_score = model.score(test_set, test_labels)
	get_true_false_positive_negative(model.predict(test_set),test_labels)
	return model, test_score


def get_true_false_positive_negative(y_pred, y):
	TP, FP, TN, FN = 0, 0, 0, 0
	if len(y_pred) != len(y):
		print('WARNING: y_pred and y are of different length:', len(y), len(y_pred))
		return 0, 0, 0, 0
	for idx, prediction in enumerate(y_pred):
		if prediction == y[idx]:
			if prediction == 1:
				TP += 1
			elif prediction == 0:
				TN += 1
		else:
			if prediction == 1:
				FP +=1
			elif prediction == 0:
				FN +=1
	precision = '-' if (TP+FP) == 0 else TP/(TP+FP)
	recall = '-' if (TP+FN) == 0 else TP/(TP+FN)
	print('TP, TN, FP, FN: ', TP, TN, FP, FN,'precision:', precision, 'recall', recall)
	return TP, TN, FP, FN, precision, recall

def _load_models_dict_list(goids):
	model_dir = RUNNING_MODEL_DIR
	models_file = model_dir+'/all_models.p'
	print('Loading models from goid list ')
	models_dict = {}
	for gid in goids:
		model_file = model_dir+'/model-'+gid.split(':')[1]+'.p'
		name = 'GO:'+model_file.split('/')[-1].split('-')[-1].split('.')[0]
		model = pickle.load(open(model_file,'rb'))
		models_dict[name] = model
	return models_dict

def _load_models_dict_subtree():
	return _load_models_dict_list(goids=_subtree_labels())


def _load_models_dict(hash='*'):
	model_dir = RUNNING_MODEL_DIR
	models_file = model_dir+'/all_models.p'
	print('Loading models from individual files')
	models_dict = {}
	for model_file in glob.glob(model_dir+'/model-*'+str(hash)+'.p'):
		name = 'GO:'+model_file.split('/')[-1].split('-')[-1].split('.')[0]
		model = pickle.load(open(model_file,'rb'))
		models_dict[name] = model
	return models_dict

def get_model_cpds(labels_list=None,modelsdir=MODEL_CPDS_DIR,use_state_names=False):
	cpds = []
	for file_name in glob.glob(modelsdir+'/*'):
		if (labels_list==None) or ('GO:'+file_name.split('/')[-1].split('.')[0] in labels_list):
			print(file_name)
			cpd = pickle.load(open(file_name,'rb'))
			if not use_state_names:
				cpd = TabularCPD(cpd.variable,cpd.variable_card,cpd.values,cpd.variables[1:],cpd.cardinality[1:])
			cpds.append(cpd)
	return cpds

def make_model_cpds_subtree(training_data,training_labels_negs,go_dict,use_state_names=False,outputdir=MODEL_CPDS_STATES_DIR):
	print('making model cpds for subtree only')
	cpds = []
	labels_list = _subtree_labels()
	for label in labels_list:
		hash = label.split(':')[-1]
		cpds += make_model_cpds(training_data,training_labels_negs,go_dict,hash,use_state_names,outputdir)
	return cpds

def safe_div(x,y):
	if y==0: return float(x+1)/(y+2)
	else: return float(x)/y

def make_model_cpds(training_data,training_labels_negs,go_dict,hash='*',subtree=False,use_state_names=False,outputdir=MODEL_CPDS_DIR):
	print('Getting cpds from svm predictions on test set')
	training_labels = np.copy(training_labels_negs)
	training_labels[np.where(training_labels==-1)] = 0
	assert((training_labels!=-1).any())
	
	if subtree:
		models_dict = _load_models_dict_subtree()
	else:
		models_dict = _load_models_dict(hash=hash)
	
	cpds = []
	i=1
	num_models = len(models_dict.keys())
	for model_name in models_dict:
		print(i,'/',num_models,' ',model_name)
		i+=1
		model = models_dict[model_name]
		if model == None:
			print('NO DATA for ',model_name,', skipping...')
			continue
		validation_data, validation_labels, _ = make_test_set(training_data,training_labels[:,go_dict[model_name]],rstate=RAND_STATE2)
		validation_labels = validation_labels.flatten()
		y_pred = model.predict(validation_data)
		yhat0_y0 = safe_div((1-y_pred[validation_labels==0]).sum(),(1-validation_labels).sum())
		yhat0_y1 = safe_div((1-y_pred[validation_labels==1]).sum(),validation_labels.sum())
		yhat1_y0 = safe_div((y_pred[validation_labels==0]).sum(),(1-validation_labels).sum())
		yhat1_y1 = safe_div((y_pred[validation_labels==1]).sum(),validation_labels.sum())
		if use_state_names:
			# state_names = {model_name+'_hat':'0',model_name+'_hat':'1'}
			state_names = [0,1]
			cpd = TabularCPD(model_name+'_hat',2,[[yhat0_y0,yhat0_y1],[yhat1_y0,yhat1_y1]],evidence=[model_name],evidence_card=[2],state_names=state_names)
			# outputdir = MODEL_CPDS_STATES_DIR
		else:
			cpd = TabularCPD(model_name+'_hat',2,[[yhat0_y0,yhat0_y1],[yhat1_y0,yhat1_y1]],evidence=[model_name],evidence_card=[2])	
			# outputdir = MODEL_CPDS_DIR
		cpd.normalize()
		cpds.append(cpd)
		if not os.path.isdir(outputdir):
			os.mkdir(outputdir)
		pickle.dump(cpd,open(outputdir+'/'+model_name.split(':')[1]+'.p','wb'))
	# pickle.dump(cpds,open(MODEL_CPDS_FILE,'wb'))
	return cpds


def get_true_label_cpds(training_labels_negs,go_dict,labels_list=None,use_state_names=False):
	print('Calculating cpds for true label dependencies')
	training_labels = np.copy(training_labels_negs)
	training_labels[np.where(training_labels==-1)] = 0	
	if labels_list == None: labels_list = go_dict.keys()
	cpds = []
	i=1
	obo_graph = obonet.read_obo(OBODB_FILE)
	for label in labels_list:
		if i % 10 == 0: print(i,'/',len(labels_list))
		i+=1
		children = [c for c in networkx.ancestors(obo_graph,label) if c in labels_list]
		data = np.full(shape=(2,2**len(children)),fill_value=0.01)
		data[1,:] = .99
		goidx = go_dict[label]
		tlc = np.copy(training_labels)
		cgidxs = [go_dict[c] for c in children]

		delete_idxs = []
		for gid in cgidxs:
			delete_idxs += list(np.where(tlc[:,gid]==1)[0])
		tlc = np.delete(tlc,delete_idxs,axis=0)
		data[1,0] = float(tlc.sum(axis=0)[goidx]) / tlc.shape[0]
		data[0,0] = 1.0 - data[1,0]

		evidence = []
		evidence_card = []
		state_names = {}
		for child in children:
			evidence.append(child)
			evidence_card.append(2)
			state_names[child] = [0,1]
		# state_names = {label+'_hat':[0,1]}
		# state_names = [0,1]
		if use_state_names:
			cpd = TabularCPD(label,2,data,evidence=evidence,evidence_card=evidence_card,state_names=state_names)
		else:
			cpd = TabularCPD(label,2,data,evidence=evidence,evidence_card=evidence_card)
		cpd.normalize()
		cpds.append(cpd)
	pickle.dump(cpds,open(TRUE_LABEL_CPDS_FILE,'wb'))
	return cpds

def _subtree_labels():
	return [l.strip() for l in open(DATA_DIR+'subtree.txt','r').readlines()]


def make_bayes_net(load=False,subtree=True,modelsdir=MODEL_CPDS_DIR):
	print('Making bayes net')
	graph_file = RUNNING_MODEL_DIR+'/'+'graph.p'
	if os.path.isfile(graph_file) and load==True:
		print('Loading saved graph from file...')
		G = pickle.load(open(graph_file,'rb'))
		G.check_model()
	else:
		print('loading data...')
		training_labels, go_dict = load_label_data()
		if subtree:
			labels_list = _subtree_labels()
			print(labels_list)
		else:
			labels_list = go_dict.keys()

		print('adding nodes and edges...')
		G = BayesianModel()
		G.add_edges_from([(label, label+'_hat') for label in labels_list])
		obo_graph = obonet.read_obo(OBODB_FILE)
		for label in labels_list:
			children = [c for c in networkx.ancestors(obo_graph,label) if c in labels_list]
			for child in children:
				G.add_edge(child,label)
		
		predicted_cpds = get_model_cpds(labels_list=labels_list,modelsdir=MODEL_CPDS_DIR)
		for cpd in predicted_cpds:
			G.add_cpds(cpd)
		true_label_cpds = get_true_label_cpds(training_labels,go_dict,labels_list=labels_list)
		for cpd in true_label_cpds:
			G.add_cpds(cpd)
		remove_list = []
		# pdb.set_trace()
		for node in G.nodes():
			if G.get_cpds(node) == None: 
				remove_list.append(node)
				# remove_list.append(node+'_hat')
		for node in remove_list:
			if node in G:
				G.remove_node(node)
		G.check_model()
		pickle.dump(G,open(graph_file,'wb'))
	return G

def make_simple_bayes_net(subtree=False):
	print('Making bayes net')
	print('loading data...')
	training_data, go_dict = load_label_data()
	if subtree:
		labels_list = _subtree_labels()
		print(labels_list)
	else:
		labels_list = go_dict.keys()

	print('adding nodes and edges...')
	G = BayesianModel()
	# G.add_edges_from([(label, label+'_hat') for label in labels_list])
	obo_graph = obonet.read_obo(OBODB_FILE)
	for label in labels_list:
		children = [c for c in networkx.ancestors(obo_graph,label) if c in labels_list]
		for child in children:
			G.add_edge(label,child)
	return G	

def MakeModelCpds(hash='*',subtree=False,use_state_names=False,outputdir=MODEL_CPDS_DIR):
	print('-------- Making model cpds -------')
	print('loading data...')
	training_data, training_labels, inv_go_dict, inv_genes_dict = LoadCombinedData()
	go_dict = {v: k for k, v in inv_go_dict.items()}
	if subtree:
		make_model_cpds_subtree(training_data,training_labels,go_dict,use_state_names,outputdir)
	else:
		make_model_cpds(training_data,training_labels,go_dict,hash,use_state_names,outputdir)


def BayesNetPredict(subtree=False,num_test_samples=1000,modelsdir=MODEL_CPDS_DIR,pred=False,prob=False):
	training_labels, go_dict = load_label_data()
	training_labels[np.where(training_labels==-1)] = 0
	model = make_bayes_net(subtree=subtree,modelsdir=modelsdir)
	test_labels = make_label_set(training_labels,rstate=RandomState(seed=RAND_SEED+2))
	bayes_nodes = model.nodes()
	
	columns = []
	idxs = []
	for goid in go_dict:
		node_name = goid+'_hat'
		if node_name in bayes_nodes: 
			columns.append(node_name)
			idxs.append(go_dict[goid])
	ftest_labels = test_labels[:,idxs]
	predict_data = pd.DataFrame(ftest_labels[0:num_test_samples],columns=columns)

	timestring = time.strftime("%m-%d-%H.%M")
	if pred:
		y_pred = model.predict(predict_data)
		pickle.dump(y_pred,open('y_pred_'+timestring+'.p','wb'))
		print(y_pred)
	if prob:
		y_prob = model.predict_probability(predict_data)
		y_true = test_labels[0:num_test_samples,:]
		pdb.set_trace()
		auc_score = metrics.roc_auc_score(y_true,y_prob)

		pickle.dump(y_prob,open('y_prob_'+timestring+'.p','wb'))
		print(y_prob)

	if pred:
		acc_dict = {}
		for c in y_pred.columns:
			y_pred_c = y_pred[c]
			true_labels = test_labels[:,go_dict[c]][0:num_test_samples]
			num_correct = (y_pred_c.values==true_labels.flatten()).sum()
			acc = float(num_correct) / num_test_samples
			acc_dict[c] = acc
			print('completed ',c,' ',acc)
		pickle.dump(acc_dict,open('acc_dict_'+timestring+'.p','wb'))
	pdb.set_trace()

def CompareSvm(subtree=False):
	print('running svms')
	y_pred_bayes = pickle.load(open('y_pred_subtree.p','rb'))
	training_data, training_labels, go_inv_dict, genes_inv_dict = LoadCombinedData()	
	go_dict = {v: k for k, v in go_inv_dict.items()}
	bayes_model = make_simple_bayes_net()
	columns = y_pred_bayes.columns
	models_dict = _load_models_dict_list(columns)
	svm_acc_dict = {}
	
	for model in models_dict:
		test_data, test_labels, test_idxs = make_test_set(training_data,training_labels[:,go_dict[model]],rstate=RAND_STATE2)
		mm = models_dict[model]
		svm_y_pred = []
		svm_y_prob = []
		test_labels_idxs = []
		for test_idx in test_idxs:
			estims = [mm.estimators_[i] for i in range(10) if test_idx not in mm.estimators_samples_[i]]
			if len(estims)>0:
				p = 0.0
				for est in estims:
					est.probability = True
					p += est.predict_proba(training_data[test_idx:test_idx+1])
				# pp = 1 if p/len(estims) >= 0.5 else 0
				# svm_y_pred.append(pp)
				svm_y_pred.append(np.array(p).mean())
				test_labels_idxs.append(test_idx)
		with open('svm_prob_auc.txt','a+'):
			y_true = training_labels[test_labels_idxs,go_dict[model]].flatten()
			auc_score = metric.roc_auc_score(y_true,svm_y_prob)
			f.write(model+'\t'+str(auc_score)+'\n')
			print(model+'\t'+str(auc_score))

		# # svm_y_pred = mm.predict(test_data)
		# # num_correct = (np.array(svm_y_pred) == np.array(test_labels)).sum()
		# num_correct = (np.array(svm_y_pred) == training_labels[test_labels_idxs,go_dict[model]].flatten()).sum()
		# acc = float(num_correct) / len(test_labels_idxs)
		# svm_acc_dict[model] = acc
		# with open('compare_acc.txt','a+') as f:
		# 	svm_acc = acc
		# 	f.write(model+'\t'+str(svm_acc)+'\n')
		# 	print(model+'\t',svm_acc)


def DrawGraph(subtree=False):
	G = make_simple_bayes_net(subtree)
	networkx.draw_networkx(G)
	plt.show()
	pdb.set_trace()

def MakeVisualizationGraph():
	G = make_bayes_net()
	remove_list = []
	for node in G.nodes():
		if '_hat' in node:
			remove_list.append(node)
	for node in remove_list:
		G.remove_node(node)
	pickle.dump(G,open('vis_graph.p','wb'))


parser = argparse.ArgumentParser()
parser.add_argument('--bags',default=1,type=int)
parser.add_argument('--startID',default=0,type=int)
parser.add_argument('--restart',action='store_true',default=False)
parser.add_argument('--cpdhash',default='*')
parser.add_argument('--subtree',action='store_true',default=False)
parser.add_argument('--statenames',action='store_true',default=False)
parser.add_argument('--outputdir',default=None)
parser.add_argument('--pred',action='store_true',default=False)
parser.add_argument('--prob',action='store_true',default=False)
options, action = parser.parse_known_args()
action = action[0]
if action == 'ma_parsegds':
	GDSFiles_to_Dict()
elif action == 'oldids':
	FixOutOfDataGoIds()
elif action == 'godict':
	MakeGeneAndGoDicts()
elif action == 'go_makearr':
	ConstructGoAnnotationArray()
elif action == 'ma_makearr':
	ConstructMicroarrayArray()
elif action == 'pair_makearr':
	ConstructPairwiseArray()
elif action == 'runsvm':
	DefaultParametersFullData(n_estimators=options.bags,startID=options.startID,resume=(not options.restart))
elif action == 'modelcpds':
	if options.outputdir:
		MakeModelCpds(hash=options.cpdhash,subtree=options.subtree,use_state_names=options.statenames,outputdir=options.outputdir)
	else:
		MakeModelCpds(hash=options.cpdhash,subtree=options.subtree,use_state_names=options.statenames)		
elif action == 'makebayes':
	make_bayes_net(load=(not options.restart))
elif action == 'runbayes':
	BayesNetPredict(subtree=options.subtree,pred=options.pred,prob=options.prob)
elif action == 'drawgraph':
	# MakeVisualizationGraph()
	DrawGraph(subtree=options.subtree)
elif action == 'compare':
	CompareSvm()
# elif action == 'bayesacc':
# 	bayes_accuracy()
else: print("missing required arguments")





