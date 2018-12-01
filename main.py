import networkx
import obonet
import csv
import numpy as np
from sklearn import svm, ensemble
import pickle

# Program can be viewed as two steps:
# 1) Build the list of examples, to go and retrieve the matching labels on https://www.uniprot.org/uploadlists/
# 2) Build a model
# Run it once with first_step = True, and then once you updated entrez_go_type.tab,
# run a second time with first_step = False
first_step = False
data_file_type = 'yeast'  # atm, should be one of ['yeast', 'type']

optimisation_on_a_single_go_id = False
output_model_dic = True

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


def get_obo_graph(dir, restrict_go_nodes_list):
    obo_graph = obonet.read_obo(dir)
    alt_ids_dic = {}
    for idx, node in obo_graph.nodes(data=True):
        if 'alt_id' in node:
            if type(node['alt_id']) == list:
                for alt_id in node['alt_id']:
                    alt_ids_dic[alt_id] = idx
            else:
                alt_ids_dic[node['alt_id']] = idx
    if restrict_go_nodes_list:
        go_nodes_list = import_go_nodes_list()
        curated_go_nodes_list = set()
        for node in go_nodes_list:
            if node in alt_ids_dic:
                curated_go_nodes_list.add(alt_ids_dic[node])
            elif node in obo_graph.nodes():
                curated_go_nodes_list.add(node)
        # TAKES TOO LONG; BETTER WORK WITH WHOLE GRAPH
        # Refactor graph to reconnect nodes linked to deleted nodes.
        # nodes_to_remove = []
        # for node in obo_graph.nodes().keys():
        #     if node not in curated_go_nodes_list \
        #             and len(obo_graph._pred[node]) > 0 and len(obo_graph._succ[node]) > 0:
        #         for incoming_node in obo_graph._pred[node]:
        #             for outgoing_node, value in obo_graph._succ[node].items():
        #                 obo_graph.add_edge(incoming_node, outgoing_node, key=list(value)[0])
        #         nodes_to_remove.append(node)
        # obo_graph.remove_nodes_from(curated_go_nodes_list)
    return obo_graph, alt_ids_dic, curated_go_nodes_list


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


def get_max_go_id(labels, go_ids_list):
    go_id_count = {}
    for go_labels in labels:
        for go_label in go_labels:
            go_id_count[go_label] = 1 if go_label not in go_id_count.keys() else go_id_count[go_label] + 1
    max_occurences = max(go_id_count.values())
    print('max_go_index_occurences:', max_occurences)
    key = -1
    for go_id in go_id_count:
        if go_id_count[go_id] == max_occurences:
            key = go_id
    for go_id_index, go_id in enumerate(list(go_ids_list)):
        if go_id == key:
            return go_id_index, key
    return -1


def get_true_false_positive_negative(y_pred, y):
    TP, FP, TN, FN = 0, 0, 0, 0
    if len(y_pred) != len(y):
        print('WARNING: y_pred and y are of different length:', len(y), len(y_pred))
        return 0, 0, 0, 0
    for idx, prediction in enumerate(y_pred):
        if prediction == y[idx]:
            if prediction == 1:
                TP += 1
            elif prediction == -1:
                TN += 1
        else:
            if prediction == 1:
                FP +=1
            elif prediction == -1:
                FN +=1
    return TP, TN, FP, FN, '-' if (TP+FP) == 0 else TP/(TP+FP), '-' if (TP+FN) == 0 else TP/(TP+FN)


def test_svm_model(kernel, training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto'):
    print('Kernel:', kernel, ',gamma:', gamma)
    model = ensemble.BaggingClassifier(svm.SVC(kernel=kernel, gamma=gamma, random_state=0), max_samples=0.632)
    model.fit(training_examples, training_labels)
    print('dev score: ', model.score(dev_set, dev_labels))
    print('test score: ', model.score(test_set, test_labels))
    true_false_positive_negative = get_true_false_positive_negative(model.predict(dev_set), dev_labels)
    print('TP, TN, FP, FN, dev: ', true_false_positive_negative[:4],
          'precision:', true_false_positive_negative[4], 'recall', true_false_positive_negative[5])
    true_false_positive_negative = get_true_false_positive_negative(model.predict(test_set), test_labels)
    print('TP, TN, FP, FN, test: ', true_false_positive_negative[:4],
          'precision:', true_false_positive_negative[4], 'recall', true_false_positive_negative[5])
    return model


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


def import_go_nodes_list():
    go_nodes_list = []
    with open('go_nodes_list.txt') as file:
        for line in file:
            go_nodes_list.append('GO:' + line.strip().rjust(7, '0'))
    return go_nodes_list


def model_data(axis, grid_data, graph, alt_ids, data_type, curated_go_nodes_list):
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
    with open('interaction_matrix', 'wb+') as fp:
        pickle.dump(interaction_matrix, fp)
    with open('matrix_index', 'wb+') as fp:
        pickle.dump(matrix_index, fp)

    # build set of all the GO IDs found in entrez_go_type.tab,
    # as well as labels which is a replica of entrez_go_type.tab colums 2 and 3
    print('-----Matching data label-----')
    go_id_set = set()
    entrez_to_go_labels = [[] for _ in range(m)]
    with open('entrez_go_' + data_type + '.tab') as entrez_go_file:
        reader = csv.DictReader(entrez_go_file, delimiter="	")
        header = reader.fieldnames
        entrez_id_key = header[1]
        for line in reader:
            if line[header[2]] == '':
                continue
            # If we have the same protein under different codes, uniprot.org will put them on the same line with a comma
            # --> we remove the duplicate entry
            if ',' in line[entrez_id_key]:
                line[entrez_id_key] = line[entrez_id_key].split(',')[0]
            for go_id in line[header[2]].split("; "):
                if go_id in curated_go_nodes_list and\
                        (go_id in graph.nodes() or (go_id in alt_ids and alt_ids[go_id] in graph.nodes())):
                    entrez_to_go_labels[matrix_index[line[entrez_id_key]]].append(go_id)
                    go_id_set.add(line[header[2]])

    # Build subsets of go_nodes and descendants
    go_ids_and_under = {}
    for node in curated_go_nodes_list:
        go_ids_and_under[node] = networkx.ancestors(graph, node)
        go_ids_and_under[node].add(node)
    # Label data
    go_id_specific_labels = np.full((len(curated_go_nodes_list), m), -1)
    for go_node_index, go_node in enumerate(curated_go_nodes_list):
        for idx, training_example_labels in enumerate(entrez_to_go_labels):
            for go_id_label in training_example_labels:
                if go_id_label in go_ids_and_under[go_node] \
                        or (go_id_label in alt_ids and alt_ids[go_id_label] in go_ids_and_under[go_node]):
                    go_id_specific_labels[go_node_index][idx] = 1
                    break
        print("positive examples ratio for node " + go_node + ":", sum(go_id_specific_labels[go_node_index] > 0),
              "/", len(go_id_specific_labels[go_node_index]))


    print('-----Modeling data-----')
    # Prepare train/dev/test sets
    training_set_limit = int(len(interaction_matrix) * 0.6)
    dev_set_limit = int(len(interaction_matrix) * 0.8)
    training_examples = [interaction_matrix[i] for i in range(training_set_limit)]
    training_labels_all_go_ids = go_id_specific_labels[:,:training_set_limit]
    dev_set = [interaction_matrix[i] for i in range(training_set_limit, dev_set_limit)]
    dev_labels_all_go_ids = go_id_specific_labels[:,training_set_limit:dev_set_limit]
    test_set = [interaction_matrix[i] for i in range(dev_set_limit, len(interaction_matrix))]
    test_labels_all_go_ids = go_id_specific_labels[:,dev_set_limit:]

    # Models comparison
    if optimisation_on_a_single_go_id:
        # Get the most represented go_label
        print('-----Accuracy for different model parameters-----')
        go_id_index, go_id = get_max_go_id(entrez_to_go_labels, curated_go_nodes_list)
        print('go_id:', go_id)
        training_labels, dev_labels, test_labels = training_labels_all_go_ids[go_id_index], dev_labels_all_go_ids[go_id_index],\
                                                   test_labels_all_go_ids[go_id_index]
        test_svm_model('linear', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels)
        test_svm_model('rbf', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto')
        test_svm_model('rbf', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='scale')
        test_svm_model('poly', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto')
        test_svm_model('poly', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='scale')
        test_svm_model('sigmoid', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='auto')
        test_svm_model('sigmoid', training_examples, training_labels, dev_set, dev_labels, test_set, test_labels, gamma='scale')

    # Computing models for each node
    if output_model_dic:
        print('-------Fitting all go nodes models---')
        model_dic = {}
        for go_id_index, go_id in enumerate(curated_go_nodes_list):
            model_dic[go_id] = ensemble.BaggingClassifier(svm.SVC(kernel='linear',random_state=0), max_samples=0.632)\
                                       .fit(training_examples, training_labels_all_go_ids[go_id_index])
            print(go_id_index + 1, '/', len(curated_go_nodes_list))
        return model_dic
    return {}
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
    graph, alt_ids, curated_go_nodes_list = get_obo_graph('go-basic.obo', True)
    if not first_step:
        model_dic = model_data(axis, grid_data, graph, alt_ids, data_file_type, curated_go_nodes_list)
        #TODO: use model_dic
