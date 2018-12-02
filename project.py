import pdb, glob, pickle, sys, os
import numpy as np
from sklearn.neighbors import NearestNeighbors
# from shlex import quote
import networkx
import obonet
import subprocess

UNIGENE_LIST_FILE = 'unigene_list.txt'
MICROARRAY_SERIES = ['GSM992', 'GSM1000', 'GSM993', 'GSM994', 'GSM995', 'GSM996', 'GSM998', 'GSM1004', 'GSM1005', 'GSM1006', 'GSM1008', 'GSM1012', 'GSM1015', 'GSM1007', 'GSM1009', 'GSM1013', 'GSM1014', 'GSM1105', 'GSM1100', 'GSM1101', 'GSM1104', 'GSM895', 'GSM1106', 'GSM1107', 'GSM1102', 'GSM1103', 'GSM1111', 'GSM899', 'GSM1041', 'GSM1047', 'GSM1042', 'GSM1043', 'GSM1044', 'GSM1045', 'GSM1046', 'GSM1055', 'GSM1029', 'GSM1030', 'GSM1032', 'GSM1033', 'GSM1034', 'GSM1048', 'GSM1049', 'GSM1050', 'GSM1051', 'GSM1052', 'GSM1053', 'GSM1054', 'GSM1075', 'GSM1076', 'GSM1090', 'GSM1077', 'GSM1078', 'GSM883', 'GSM930', 'GSM929', 'GSM928', 'GSM926', 'GSM925', 'GSM854', 'GSM855', 'GSM856', 'GSM857', 'GSM864', 'GSM865', 'GSM868', 'GSM872', 'GSM1002', 'GSM1003', 'GSM842', 'GSM843', 'GSM844', 'GSM845', 'GSM846', 'GSM847', 'GSM848', 'GSM849', 'GSM850', 'GSM851', 'GSM880', 'GSM881', 'GSM882', 'GSM874', 'GSM875', 'GSM876', 'GSM877', 'GSM878', 'GSM879', 'GSM972', 'GSM1039', 'GSM1040', 'GSM1037', 'GSM938', 'GSM939', 'GSM907', 'GSM990', 'GSM991', 'GSM997', 'GSM999', 'GSM1001', 'GSM971', 'GSM1057', 'GSM1058', 'GSM1059', 'GSM1060', 'GSM1061', 'GSM1063', 'GSM1064', 'GSM961', 'GSM962', 'GSM963', 'GSM964', 'GSM965', 'GSM966', 'GSM967', 'GSM968', 'GSM1019', 'GSM1020', 'GSM1021', 'GSM1022', 'GSM1023', 'GSM934', 'GSM935', 'GSM936', 'GSM1025', 'GSM937', 'GSM1024', 'GSM918', 'GSM919', 'GSM932', 'GSM933', 'GSM980', 'GSM863', 'GSM921', 'GSM920', 'GSM988', 'GSM922', 'GSM989', 'GSM858', 'GSM902', 'GSM931', 'GSM861', 'GSM862', 'GSM923', 'GSM860', 'GSM924', 'GSM859', 'GSM940', 'GSM942', 'GSM910', 'GSM969', 'GSM970', 'GSM973', 'GSM974', 'GSM975', 'GSM976', 'GSM984', 'GSM977', 'GSM903', 'GSM906', 'GSM985']
# GO_NODES = open('go_nodes_list.txt').readlines()
GO_NODES_LIST_FILE = 'go_nodes_list.txt'
GENES_LIST_FILE = 'final_genes_list.txt'
# MA_GENES_LIST_FILE = 'microarray_genes_list.txt'
GO_DICT_FILE = 'go_dict.p'
GENES_DICT_FILE = 'genes_dict.p'
GO_ANNOTATION_FILE = 'uniprot-GO-annotations.txt'
GO_GENES_LIST_FILE = 'go_genes_list.txt'
GO_LABEL_ARR_FILE = 'go_label_arr.np'
MA_RAW_DICT_FILE = 'microarray_raw_dict.p'
PAIR_DICT_FILE ='matrix_index.txt'
MA_DICT_FILE = 'microarray_dict.p'

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


# # Discard microarray genes where entire experiments / feature sets are missing
# # N.B. This does not discard genes where experiment results were recorded as null
# def DiscardMAGenesWithMissingFeatures():
# 	data_dict = pickle.load(open('microarray_dict.p','rb'))
# 	ma_genes_list = []
# 	with open('microarray_genes_list.txt','w') as f:
# 		for gene in data_dict.keys():
# 			if set(list(data_dict[gene].keys())) == set(MICROARRAY_SERIES):
# 				ma_genes_list.append(gene)
# 				f.write(gene+'\n')
# 	return ma_genes_list

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
	print("Writing MA index dict to ")

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
	np.savetxt('microarray_with_nulls.np',data_arr)
	print("Removing nulls with Knn")
	RunKnnOnNulls()

def RunKnnOnNulls():
	data_arr = np.loadtxt('microarray_with_nulls.txt')
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
		non_null_data = smoothed_data[np.where(~np.isnan(data_arr[:,j]))]
		model = NearestNeighbors(n_neighbors=16).fit(non_null_data)
		models.append(model)
	
	# iterate through training examples and replace null values with
	# value from that feature of nearest centroid
	for i in np.where(null_axes==True)[0]:
		print(i)
		example = np.copy(data_arr[i])
		nan_indices = []
		for j in range(f):
			if np.isnan(example[j]):
				nan_indices.append(j)
				example[j] = feature_averages[j]
		if np.isnan(example).any():
			pdb.set_trace()
		for j in nan_indices:
			distances, indices = models[j].kneighbors(example.reshape(1,-1))
			centroid = np.average(data_arr[indices.flatten()],axis=0,weights=(1/distances.flatten()))
			data_arr[i,j] = centroid[j]
	np.savetxt('microarray_nonnull.txt')

def MakeGeneAndGoDicts():
	# make dict of index-to-GOTerm, using set 105 terms
	go_dict = {}
	with open(GO_NODES_LIST_FILE,'r') as f:
		lines = f.readlines()
		for i in range(len(lines)):
			go_dict[lines[i]] = i
	pickle.dump(go_dict,open(GO_DICT_FILE, "wb"))
	
	# make dict of index-to-GeneId, using intersection of GO annotations, pairwise data, and microarray data
	genes_dict = {}
	MA_genes = set(pickle.load(open(MA_RAW_DICT_FILE,'rb')).keys())
	PAIR_genes = set(pickle.load(open(PAIR_DICT_FILE,'rb')).keys())
	os.system("cat "+GO_ANNOTATION_FILE+"| tail -n+2 | awk '{{ print $1 }}' > "+GO_GENES_LIST_FILE)
	GO_genes = set([line.strip() for line in open(GO_GENES_LIST_FILE,'r').readlines()])
	pdb.set_trace()
	i = 0
	for gene in MA_genes:
		if (gene in GO_genes) and (gene in PAIR_genes):
			genes_dict[gene] = i
			i+=1
	pickle.dump(genes_dict,open(GENES_DICT_FILE,'wb'))
	print("MA_genes: "+str(len(MA_genes))+", GO_genes:"+str(len(GO_genes))+", PAIR_genes:"+str(len(PAIR_genes)))
	print("Found "+str(len(genes_dict.keys()))+" intersecting genes across all sets")


def ConstructGoAnnotationArray():
	# genes_list = open(GENES_LIST_FILE,'r'),readlines()
	# go_nodes = open(GO_NODES_LIST_FILE,'r').readlines()
	genes_dict = pickle.load(open(GENES_DICT_FILE,'rb'))
	go_dict = pickle.load(open(GO_DICT_FILE,'rb'))
	m = len(genes_dict.keys())
	f = len(go_dict.keys())
	go_labels = np.zeros(shape=(m,f))
	
	lines = open(GO_ANNOTATION_FILE,'r').readlines()
	for x in range(1,len(lines)):
		line = lines[x]
		values = line.split('\t')
		geneids = values[0].split(',')
		for geneid in geneids:
			if geneid in genes_dict.keys():
				goids = [v.strip('GO:').strip(';') for v in values[2].split(' ')]
				gene_idx = genes_dict[geneid]
				for go_id in goids:
					if go_id in go_dict:
						go_idx = go_dict[go_id]
						go_labels[gene_idx,go_idx] = 1
	np.savetxt(GO_LABEL_ARR_FILE,go_labels)
	print("Constructed go annotation array with "+str(m)+" examples and "+str(f)+" annotations")

def CombineWithPairwiseData():
	my_genes_list = [line.strip() for line in open(GENES_LIST_FILE).readlines()]
	ben_genes_dict = pickle.load(open('matrix_index.txt','rb'))
	# pdb.set_trace()
	for gene in my_genes_list:
		if gene not in ben_genes_dict.keys():
			print(gene)
	print("only pairwise:")
	for gene in ben_genes_dict.keys():
		if gene not in my_genes_list:
			print(gene)

def quote(s):
    return "'" + s.replace("'", "'\\''") + "'"



if sys.argv[1] == 'ma_parsegds':
	GDSFiles_to_Dict()
# elif sys.argv[1] == 'ma_filter':
# 	DiscardMAGenesWithMissingFeatures()
elif sys.argv[1] == 'godict':
	MakeGeneAndGoDicts()
elif sys.argv[1] == 'go_makearr':
	ConstructGoAnnotationArray()
elif sys.argv[1] == 'ma_makearr':
	ConstructMicroarrayArray()
# elif sys.argv[1] == 'ma_knn':
# 	RunKnnOnNulls()

elif sys.argv[1] == 'combinepairwise':
	CombineWithPairwiseData()
else: print("missing required arguments")





