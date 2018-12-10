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

RAND_SEED=0
RAND_STATE=np.random.RandomState(seed=RAND_SEED)
RAND_STATE2=np.random.RandomState(seed=RAND_SEED+1)
DATA_DIR =r'data/'
OUTPUT_DIR=r'output/'
RESULTS_DIR=r'results/'
RUNNING_MODEL_DIR = RESULTS_DIR+'1-running_model'
MODEL_CPDS_FILE = RUNNING_MODEL_DIR+'/model_cpds.p'
MODEL_CPDS_DIR = RUNNING_MODEL_DIR+'/model_cpds'
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
	genes_dict = pickle.load(open(GENES_DICT_FILE,'rb'))
	return label_data, go_dict, {v: k for k, v in go_dict.items()}, {v: k for k, v in genes_dict.items()}

# def _modelfnames(model_name='defaultparams'):
# 	timestring = time.strftime("%m-%d-%H.%M")
# 	scores_file = RESULTS_DIR+model_name+'_'+timestring+'_scores.txt'
# 	models_file = RESULTS_DIR+model_name+'_'+timestring+'_model_'
# 	return scores_file, models_file

# def _modelDir():
# 	os.mkdir(RUNNING_MODEL_DIR)
# 	return RUNNING_MODEL_DIR

def _timeModelDir(model_name='lineardefault'):
	timestring = time.strftime("%m-%d-%H.%M")
	dirname = RESULTS_DIR+model_name+'_'+timestring
	os.mkdir(dirname)
	return dirname


def DefaultParametersFullData(kernel='linear',C=1.0,gamma='auto',n_estimators=1,resume=True,startID=0):
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
	
	for i in range(startID,training_labels.shape[1]):
		goid = go_inv_dict[i]
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
	os.rename(model_dir,_timeModelDir(model_name='lineardefault'))


def make_test_set(training_examples,training_labels,rstate=RAND_STATE):
	num_test_samples = int(training_examples.shape[0]*0.632)
	test_idxs = rstate.choice(range(0,training_examples.shape[0]),size=num_test_samples,replace=True)
	test_set = np.zeros(shape=(num_test_samples,training_examples.shape[1]))
	test_labels = np.zeros(shape=(num_test_samples,1))
	for i in range(0,num_test_samples):
		test_set[i] = training_examples[test_idxs[i]]
		test_labels[i] = training_labels[test_idxs[i]]
	return test_set, test_labels


def test_svm_model(kernel, training_examples, training_labels, C=1.0, gamma='auto',n_estimators=10):
	model = ensemble.BaggingClassifier(svm.SVC(kernel=kernel, gamma=gamma, random_state=RAND_SEED), n_estimators=n_estimators,max_samples=0.632)
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

def _dumps(models_dict,file_path):
	n_bytes = 2**31
	max_bytes = n_bytes - 1
	## write
	bytes_out = pickle.dumps(models_dict)
	with open(file_path, 'wb') as f_out:
		for idx in range(0, len(bytes_out), max_bytes):
			f_out.write(bytes_out[idx:idx+max_bytes])	

def _loads(file_path):
	n_bytes = 2**31
	max_bytes = n_bytes - 1
	## read
	bytes_in = bytearray(0)
	input_size = os.path.getsize(file_path)
	with open(file_path, 'rb') as f_in:
		for _ in range(0, input_size, max_bytes):
			bytes_in += f_in.read(max_bytes)
	data2 = pickle.loads(bytes_in)
	return data2


def _load_models_dict(hash='*'):
	model_dir = RUNNING_MODEL_DIR
	models_file = model_dir+'/all_models.p'
	print('Loading models from individual files')
	models_dict = {}
	for model_file in glob.glob(model_dir+'/model-*'+str(hash)+'.p'):
		name = 'GO:'+model_file.split('/')[-1].split('-')[-1].split('.')[0]
		model = pickle.load(open(model_file,'rb'))
		models_dict[name] = model
	# _dumps(models_dict,models_file)	
	# if not os.path.isfile(models_file):
	# 	print('Loading models from individual files')
	# 	models_dict = {}
	# 	for model_file in glob.glob(model_dir+'/model-*.p'):
	# 		name = 'GO:'+model_file.split('/')[-1].split('-')[-1].split('.')[0]
	# 		model = pickle.load(open(model_file,'rb'))
	# 		models_dict[name] = model
	# 	_dumps(models_dict,models_file)
	# 	# pickle.dump(models_dict,open(models_file,'wb'))
	# else:
	# 	print('Loading models from single file')
	# 	# models_dict = pickle.load(open(models_file,'rb'))
	# 	models_dict = _loads(models_file)
	return models_dict

def get_model_cpds():
	pass

def make_model_cpds(training_data,training_labels_negs,go_dict,hash='*'):
	print('Getting cpds from svm predictions on test set')
	# training_data, training_labels, go_inv_dict, genes_inv_dict = LoadCombinedData()
	training_labels = np.copy(training_labels_negs)
	training_labels[np.where(training_labels==-1)] = 0
	assert((training_labels!=-1).any())
	models_dict = _load_models_dict(hash=hash)
	cpds = []
	i=1
	num_models = len(models_dict.keys())
	if not os.path.isdir(MODEL_CPDS_DIR):
		os.mkdir(MODEL_CPDS_DIR)
	for model_name in models_dict:
		print(i,'/',num_models,' ',model_name)
		i+=1
		validation_data, validation_labels = make_test_set(training_data,training_labels[:,go_dict[model_name]],rstate=RAND_STATE2)
		validation_labels = validation_labels.flatten()
		model = models_dict[model_name]
		y_pred = model.predict(validation_data)
		yhat0_y0 = (1-y_pred[validation_labels==0]).sum() / (1-validation_labels).sum()
		yhat0_y1 = (1-y_pred[validation_labels==1]).sum() / validation_labels.sum()
		yhat1_y0 = (y_pred[validation_labels==0]).sum() / (1-validation_labels).sum()
		yhat1_y1 = (y_pred[validation_labels==1]).sum() / validation_labels.sum()
		cpd = TabularCPD(model_name+'_hat',2,[[yhat0_y0,yhat0_y1],[yhat1_y0,yhat1_y1]],evidence=[model_name],evidence_card=[2])
		cpd.normalize()
		cpds.append(cpd)
		pickle.dump(cpd,open(MODEL_CPDS_DIR+'/'+model_name.split(':')[1]+'.p','wb'))
	pickle.dump(cpds,open(MODEL_CPDS_FILE,'wb'))
	return cpds


def get_true_label_cpds(training_labels_negs,go_dict):
	print('Calculating cpds for true label dependencies')
	training_labels = np.copy(training_labels_negs)
	training_labels[np.where(training_labels==-1)] = 0	
	labels_list = go_dict.keys()
	cpds = []
	i=1
	obo_graph = obonet.read_obo(OBODB_FILE)
	for label in labels_list:
		print(i,'/',len(labels_list))
		children = [c for c in networkx.ancestors(obo_graph,label) if c in labels_list]
		data = np.zeros(shape=(2,2**len(children)))
		data[1,:] = 1
		goidx = go_dict[label]
		tlc = np.copy(training_labels)
		cgidxs = [go_dict[c] for c in children]
		for gid in cgidxs:
			np.delete(tlc,np.where(tlc[:,gid]==1))
		data[1,0] = tlc.sum(axis=0)[goidx] / tlc.shape[0]
		data[0,0] = 1 - data[1,0]

		evidence = []
		evidence_card = []
		for child in children:
			evidence.append(child)
			evidence_card.append(2)
		cpd = TabularCPD(label,2,data,evidence=evidence,evidence_card=evidence_card)
		cpds.append(cpd)
	pickle.dump(cpds,open(TRUE_LABEL_CPDS_FILE,'wb'))
	return cpds


def make_bayes_net():
	print('Making bayes net')
	print('loading data...')
	training_data, training_labels, inv_go_dict, inv_genes_dict = LoadCombinedData()
	go_dict = {v: k for k, v in inv_go_dict.items()}
	labels_list = go_dict.keys()

	print('adding nodes and edges...')
	G = BayesianModel()
	# G.add_nodes_from(labels_list)
	# G.add_nodes_from([label+'_hat' for label in labels_list])
	G.add_edges_from([(label, label+'_hat') for label in labels_list])
	obo_graph = obonet.read_obo(OBODB_FILE)
	for label in labels_list:
		children = [c for c in networkx.ancestors(obo_graph,label) if c in labels_list]
		for child in children:
			G.add_edge(child,label)
			# cpd = TabularCPD(label,2,[[1,0],[0,1]],evidence=[child],evidence_card=[2])
			# G.add_cpds(cpd)
	
	predicted_cpds = get_model_cpds(training_data,training_labels,go_dict)
	for cpd in predicted_cpds:
		G.add_cpds(cpd)
	true_label_cpds = get_true_label_cpds(training_labels,go_dict)
	for cpd in true_label_cpds:
		G.add_cpds(cpd)
	pickle.dump(G,open(RUNNING_MODEL_DIR+'/'+'graph.p'))
	return G

def MakeModelCpds(hash='*'):
	print('-------- Making model cpds -------')
	print('loading data...')
	training_data, training_labels, inv_go_dict, inv_genes_dict = LoadCombinedData()
	go_dict = {v: k for k, v in inv_go_dict.items()}
	make_model_cpds(training_data,training_labels,go_dict,hash=hash)
	
def BayesNetPredict():
	training_data, training_labels, go_inv_dict, genes_inv_dict = LoadCombinedData()
	model = make_bayes_net()
	test_data, test_labels = make_test_set(training_data,training_labels,rstate=RandomState(seed=RAND_SEED+2))
	columns = [go_inv_dict[i]+'_hat' for i in range(len(go_inv_dict.keys()))]
	predict_data = pd.DataFrame(test_labels[0],columns=columns)
	y_pred = model.predict(predict_data)
	pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('--bags',default=1,type=int)
parser.add_argument('--startID',default=0,type=int)
parser.add_argument('--restart',action='store_true',default=False)
parser.add_argument('--cpdhash',default='*')
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
	MakeModelCpds(hash=options.cpdhash)
elif action == 'runbayes':
	BayesNetPredict()
else: print("missing required arguments")





