import networkx
import obonet
import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
import json
import pickle

# Program can be viewed as two steps:
# 1) Build the list of examples, to go and retrieve the matching labels on https://www.uniprot.org/uploadlists/
# 2) Build a model
# Run it once with first_step = True, and then once you updated entrez_go_type.tab,
# run a second time with first_step = False
first_step = False
data_file_type = 'yeast'  # atm, should be one of ['yeast', 'type']

biological_processes = 'GO:0008150'
molecular_function = 'GO:0003674'
bio_grid_root_folder = 'BIOGRID-SYSTEM-3.5.166.mitab/'
biogrid_file_names = [
    # 'BIOGRID-SYSTEM-Two-hybrid-3.5.166.mitab.txt',
    'BIOGRID-SYSTEM-Synthetic_Lethality-3.5.166.mitab.txt',
    'BIOGRID-SYSTEM-Synthetic_Rescue-3.5.166.mitab.txt',
    'BIOGRID-SYSTEM-Dosage_Lethality-3.5.166.mitab.txt'
    ]
biogrid_yeast_specific = ['BIOGRID-ORGANISM-3.5.166.mitab/'
                          'BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.166.mitab.txt']
kernels = ['linear', 'rbf']


def get_obo_graph(dir):
    obo_graph = obonet.read_obo(dir)
    alt_ids_dic = {}
    for idx, node in obo_graph.nodes(data=True):
        if 'alt_id' in node:
            for alt_id in node['alt_id']:
                alt_ids_dic[alt_id] = idx
    return obo_graph, alt_ids_dic


def parse_interaction_data(file_names):
    grid_data = []
    # Get interaction data with entrez ID (list of [Interactor A ID, Interactor B ID]
    for filename in file_names:
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='	')
            i = 0
            for row in reader:
                i += 1
                grid_data.append([row["#ID Interactor A"].split(':')[1],
                                  row["ID Interactor B"].split(':')[1]])
    print('-----Parsed interaction file-----')
    return grid_data


def get_max_go_id(go_id_count):
    max_occurences = max(go_id_count.values())
    print('max_go_index_occurences:', max_occurences)
    for key in go_id_count:
        if go_id_count[key] == max_occurences:
            return key
    return -1


def get_true_false_positive_negative(y_pred, y):
    TP, FP, TN, FN = 0, 0, 0, 0
    if len(y_pred) != len(y):
        print('WARNING: y_pred and y are of different length:', len(y), len(y_pred))
        return 0, 0, 0, 0
    for idx, prediction in enumerate(y_pred):
        if prediction == y[idx]:
            if prediction:
                TP += 1
            else:
                TN += 1
        else:
            if prediction:
                FP +=1
            else:
                FN +=1
    return TP, TN, FP, FN


def test_svm_model(kernel, training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto'):
    print('Kernel:', kernel)
    model = svm.SVC(kernel=kernel, gamma=gamma, random_state=0)
    model.fit(training_examples, training_labels)
    print('dev score: ', model.score(dev_set, dev_labels))
    print('test score: ', model.score(test_set, test_labels))
    print('TP, TN, FP, FN, dev: ', get_true_false_positive_negative(model.predict(dev_set), dev_labels))
    print('TP, TN, FP, FN, test: ', get_true_false_positive_negative(model.predict(test_set), test_labels))


def build_entrez_id_set(data_type):
    grid_data = parse_interaction_data([bio_grid_root_folder + biogrid_file_names[i]
                                        for i in range(len(biogrid_file_names))] if data_type == 'type'
                                       else biogrid_yeast_specific)
    # Get the set of unique IDs in our DB
    axis = set()
    for item in grid_data:
        for index in item:
            axis.add(index)
    # save set for labelling (using https://www.uniprot.org/uploadlists/).
    # The output data file from https://www.uniprot.org/uploadlists/ should have entrez & GO IDs as 2nd and 3rd columns.
    with open('entrez_id_set.txt', 'w+') as entrez_id_set:
        string = ""
        for item in axis:
            string += item + " "
        string = string[:-1]
        entrez_id_set.write(string)
    print('number of different genes:', len(axis))
    return axis, grid_data


def model_data(axis, grid_data, graph, alt_ids, data_type):
    print('Data type:', data_type)
    # build interaction matrix
    m = len(axis)
    interaction_matrix = np.zeros((m, m))
    matrix_index = {}
    i = 0
    for elem in axis:
        matrix_index[elem] = i
        i += 1
    for item in grid_data:
        interaction_matrix[matrix_index[item[0]], matrix_index[item[1]]] += 1
    # build set of all the GO IDs found in entrez_go_type.tab,
    # as well as labels which is a replica of entrez_go_type.tab colums 2 and 3
    print('-----Matching data label-----')
    go_id_set = set()
    labels = [[] for _ in range(m)]
    with open('entrez_go_' + data_type + '.tab') as entrez_go_file:
        reader = csv.DictReader(entrez_go_file, delimiter="	")
        header = reader.fieldnames
        entrez_id_key = header[1]
        for line in reader:
            # If we have the same protein under different codes, uniprot.org will put them on the same line with a coma
            # --> we remove the duplicate entry
            if line[header[2]] == '':
                continue
            if ',' in line[entrez_id_key]:
                line[entrez_id_key] = line[entrez_id_key].split(',')[0]
            for go_id in line[header[2]].split("; "):
                labels[matrix_index[line[entrez_id_key]]].append(go_id)
                go_id_set.add(line[header[2]])
    go_id_count = {}
    for go_labels in labels:
        for go_label in go_labels:
            go_id_count[go_label] = 1 if go_label not in go_id_count.keys() else go_id_count[go_label] + 1
    go_id = get_max_go_id(go_id_count)
    print('go_id:', go_id)
    go_id_specific_labels = np.full(m, -1)
    for idx, training_example_labels in enumerate(labels):
        if go_id in training_example_labels:
            go_id_specific_labels[idx] = 1
        else:
            for go_id_label in training_example_labels:
                if (go_id_label in graph.nodes() and go_id in networkx.ancestors(graph, go_id_label))\
                        or (go_id_label in alt_ids and go_id in networkx.ancestors(graph, alt_ids[go_id_label])):
                    go_id_specific_labels[idx] = 1
                    break
    print("positive examples ratio:", sum(go_id_specific_labels[i] == 1 for i in range(len(go_id_specific_labels))),
          "/", len(go_id_specific_labels))
    print('-----Modeling data-----')
    training_set_limit = int(len(interaction_matrix) * 0.6)
    dev_set_limit = int(len(interaction_matrix) * 0.8)
    training_examples = [interaction_matrix[i] for i in range(training_set_limit)]
    training_labels = go_id_specific_labels[:training_set_limit]
    dev_set = [interaction_matrix[i] for i in range(training_set_limit, dev_set_limit)]
    dev_labels = go_id_specific_labels[training_set_limit:dev_set_limit]
    test_set = [interaction_matrix[i] for i in range(dev_set_limit, len(interaction_matrix))]
    test_labels = go_id_specific_labels[dev_set_limit:]

    # Models comparison
    with open('deterministic/training_examples_debug' + data_type, 'wb+') as fp:
        pickle.dump(interaction_matrix, fp)
    with open('deterministic/training_labels_debug' + data_type, 'wb+') as fp:
        pickle.dump(go_id_specific_labels, fp)

    test_svm_model('linear', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels)
    test_svm_model('rbf', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto')
    test_svm_model('rbf', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='scale')
    test_svm_model('poly', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto')
    test_svm_model('poly', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='scale')
    test_svm_model('sigmoid', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto')
    test_svm_model('sigmoid', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='scale')

    # Need to figure out this someday.
    # for C in np.logspace(-3, 3, 7):
    #     for gamma in np.logspace(-4, 2, 7):
    #         print('Kernel: rbf, C =', C, 'gamma =', gamma)
    #         model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    #         model.fit(training_examples, training_labels)
    #         print('dev score: ', model.score(dev_set, dev_labels))
    #         print('test score: ', model.score(test_set, test_label))

    # from https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    # C_range = np.logspace(-3, 3, 7)
    # gamma_range = np.logspace(-4, 2, 7)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=229)
    # grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose=49, n_jobs=-1)
    # print('-----Created GridSearch-----')
    # grid.fit([interaction_matrix[i] for i in range(len(interaction_matrix))], go_id_specific_labels)
    # print('-----Fitting finished-----')
    # print(grid.cv_results_)
    # print("The best parameters are %s with a score of %0.2f"
          # % (grid.best_params_, grid.best_score_))


if __name__ == '__main__':
    print('-----Parsing interaction data-----')
    axis, grid_data = build_entrez_id_set(data_file_type)
    print('-----Building GO graph-----')
    graph, alt_ids = get_obo_graph('go-basic.obo')
    if not first_step:
        model_data(axis, grid_data, graph, alt_ids, data_file_type)
